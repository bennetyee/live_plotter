use chrono::{DateTime, Local, Offset};
use clap::Parser;
use eframe::egui;
use egui::{Color32, TextStyle};
use egui_plot::{Corner, GridInput, GridMark, Legend, Line, Plot, PlotBounds, PlotPoints};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "High-Performance Live Multi-Series Plotter")]
struct Args {
    /// Maximum number of data points to retain per series (ring buffer size)
    #[arg(short, long, default_value_t = 1_000_000)]
    max_points: usize,

    /// Initial number of X-axis units visible in the viewport
    #[arg(short, long, default_value_t = 500.0)]
    viewport_width: f64,

    /// Interpret the first column as a Unix timestamp (seconds since epoch)
    #[arg(short, long, default_value_t = false)]
    timestamp: bool,

    /// Fixed period in seconds between samples for non-timestamped data
    #[arg(long, default_value_t = 5.0)]
    sample_period: f64,

    /// Y-axis values that should always be visible (baseline/ceiling)
    #[arg(long, num_args = 1..)]
    include_y: Vec<f64>,

    /// Labels for the data series
    #[arg(short, long, num_args = 1..)]
    labels: Option<Vec<String>>,

    /// Sort labels alphabetically (default is command-line order)
    #[arg(long, default_value_t = false)]
    sort_labels: bool,

    /// Hex colors for the lines (e.g., #ff0000 #00ff00)
    #[arg(short, long, num_args = 1..)]
    colors: Option<Vec<String>>,

    /// Title displayed at the top of the graph
    #[arg(long, default_value = "Live Time-Series Feed")]
    title: String,

    /// Legend position: LeftTop, RightTop, LeftBottom, RightBottom, or None
    #[arg(long, default_value = "LeftTop")]
    legend_pos: String,

    /// Maximum smoothing time constant (tau) in seconds
    #[arg(long, default_value_t = 60.0)]
    max_tau: f64,

    /// Input source: "-" for stdin (default), a file path, or a Unix socket path
    #[arg(long, default_value = "-")]
    input: String,

    /// Accept newline-delimited JSON input: {"t":1234,"values":{"label":val,...}}
    /// Implies --timestamp; labels are discovered from first line if not given.
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Log all received data to this CSV file (appends if file exists)
    #[arg(long)]
    output: Option<String>,

    /// Max rendered points per series (LTTB target). Default auto-sizes to 4x pixel width.
    #[arg(long)]
    max_render_points: Option<usize>,
}

/// A ring-buffer backed series of [x, y] samples.
struct SeriesBuffer {
    data: VecDeque<[f64; 2]>,
    capacity: usize,
}

impl SeriesBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity.min(65536)),
            capacity,
        }
    }

    fn push(&mut self, point: [f64; 2]) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(point);
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Binary search on the x-axis value. Returns the index into `data`.
    fn search_x(&self, x: f64) -> usize {
        let (mut lo, mut hi) = (0usize, self.data.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.data[mid][0] < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn first(&self) -> Option<&[f64; 2]> {
        self.data.front()
    }

    fn last(&self) -> Option<&[f64; 2]> {
        self.data.back()
    }
}

// ---------------------------------------------------------------------------
// LTTB – Largest-Triangle-Three-Buckets decimation
// Returns at most `threshold` points from the slice [start..end] of `buf`,
// preserving the visual shape of the time series.
// ---------------------------------------------------------------------------
fn lttb(buf: &SeriesBuffer, start: usize, end: usize, threshold: usize) -> Vec<[f64; 2]> {
    let count = end - start;
    if threshold == 0 || count <= threshold {
        return (start..end).map(|i| buf.data[i]).collect();
    }

    let mut sampled: Vec<[f64; 2]> = Vec::with_capacity(threshold);
    // Always keep the first point.
    sampled.push(buf.data[start]);

    let bucket_size = (count - 2) as f64 / (threshold - 2) as f64;
    let mut a = 0usize; // index of last selected point (relative to start)

    for i in 0..(threshold - 2) {
        // Calculate the range for this bucket
        let b_start = (((i + 1) as f64 * bucket_size) as usize + 1).min(count - 1);
        let b_end = ((((i + 2) as f64 * bucket_size) as usize) + 1).min(count - 1);

        // Calculate the average point in the NEXT bucket (the "C" point)
        let c_start = b_end;
        let c_end = ((((i + 3) as f64 * bucket_size) as usize) + 1).min(count);
        let avg_len = (c_end - c_start).max(1) as f64;
        let (avg_x, avg_y) = (c_start..c_end).fold((0.0, 0.0), |(ax, ay), idx| {
            (ax + buf.data[start + idx][0], ay + buf.data[start + idx][1])
        });
        let (avg_x, avg_y) = (avg_x / avg_len, avg_y / avg_len);

        // Find the point in this bucket that forms the largest triangle with A and avg(C)
        let ax = buf.data[start + a][0];
        let ay = buf.data[start + a][1];
        let mut max_area = -1.0f64;
        let mut max_idx = b_start;
        for j in b_start..b_end {
            let px = buf.data[start + j][0];
            let py = buf.data[start + j][1];
            let area = ((ax - avg_x) * (py - ay) - (ax - px) * (avg_y - ay)).abs();
            if area > max_area {
                max_area = area;
                max_idx = j;
            }
        }
        sampled.push(buf.data[start + max_idx]);
        a = max_idx;
    }

    // Always keep the last point.
    sampled.push(buf.data[start + count - 1]);
    sampled
}

// ---------------------------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
enum XAnchor {
    Left,
    Center,
    Right,
}

#[derive(PartialEq, Clone, Copy)]
enum YAnchor {
    Top,
    Center,
    Bottom,
}

struct LivePlotApp {
    raw_data: Arc<Mutex<Vec<SeriesBuffer>>>,
    smoothed_data: Arc<Mutex<Vec<SeriesBuffer>>>,
    stream_ended: Arc<AtomicBool>,
    tau_shared: Arc<Mutex<f64>>,
    include_y_values: Vec<f64>,
    labels: Vec<String>,
    colors: Vec<Color32>,
    visible: Vec<bool>,
    title: String,
    legend_corner: Option<Corner>,
    is_ts_mode: bool,

    // UI State
    side_panel_collapsed: bool,
    tau: f64,
    max_tau: f64,
    default_width: f64,
    max_render_points: Option<usize>,
    view_settling: bool,

    // Viewport State
    x_zoom: f64,
    scroll_offset: f64,
    y_zoom: f64,
    y_min: f64,
    y_max: f64,
    y_nat_h: f64,
    auto_follow: bool,
    anchor_x: XAnchor,
    anchor_y: YAnchor,

    last_view_rect: Option<[f64; 4]>,
}

impl LivePlotApp {
    fn new(
        args: Args,
        raw_data: Arc<Mutex<Vec<SeriesBuffer>>>,
        smoothed_data: Arc<Mutex<Vec<SeriesBuffer>>>,
        tau_shared: Arc<Mutex<f64>>,
        stream_ended: Arc<AtomicBool>,
        labels: Vec<String>,
    ) -> Self {
        let legend_corner = match args.legend_pos.to_lowercase().as_str() {
            "none" => None,
            "lefttop" => Some(Corner::LeftTop),
            "righttop" => Some(Corner::RightTop),
            "leftbottom" => Some(Corner::LeftBottom),
            "rightbottom" => Some(Corner::RightBottom),
            _ => {
                eprintln!("FATAL ERROR: Invalid --legend-pos '{}'.", args.legend_pos);
                eprintln!("Valid values: LeftTop, RightTop, LeftBottom, RightBottom, None");
                std::process::exit(1);
            }
        };

        Self {
            raw_data,
            smoothed_data,
            stream_ended,
            tau_shared,
            include_y_values: args.include_y,
            labels: labels.clone(),
            colors: Self::generate_palette(args.colors),
            visible: vec![true; labels.len()],
            title: args.title,
            legend_corner,
            is_ts_mode: args.timestamp || args.json,
            side_panel_collapsed: false,
            tau: 0.000001,
            max_tau: args.max_tau,
            default_width: args.viewport_width,
            max_render_points: args.max_render_points,
            view_settling: false,
            x_zoom: 1.0,
            scroll_offset: 0.0,
            y_zoom: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            y_nat_h: 1.0,
            auto_follow: true,
            anchor_x: XAnchor::Right,
            anchor_y: YAnchor::Bottom,
            last_view_rect: None,
        }
    }

    fn generate_palette(user_colors: Option<Vec<String>>) -> Vec<Color32> {
        let palette = [
            Color32::from_rgb(255, 85, 85),
            Color32::from_rgb(85, 255, 85),
            Color32::from_rgb(85, 85, 255),
            Color32::from_rgb(255, 255, 85),
            Color32::from_rgb(255, 85, 255),
            Color32::from_rgb(85, 255, 255),
            Color32::from_rgb(255, 170, 0),
            Color32::from_rgb(170, 0, 255),
            Color32::from_rgb(0, 255, 170),
            Color32::from_rgb(255, 0, 127),
            Color32::from_rgb(170, 255, 0),
            Color32::from_rgb(0, 170, 255),
            Color32::from_rgb(255, 215, 180),
            Color32::from_rgb(128, 128, 128),
            Color32::from_rgb(170, 110, 40),
            Color32::from_rgb(0, 128, 128),
            Color32::from_rgb(230, 190, 255),
            Color32::from_rgb(128, 0, 0),
            Color32::from_rgb(170, 255, 195),
            Color32::from_rgb(128, 128, 0),
        ];
        let mut colors: Vec<Color32> = Vec::new();
        if let Some(h) = user_colors {
            for hex in h {
                if let Ok(c) = Color32::from_hex(&hex) {
                    colors.push(c);
                }
            }
        }
        while colors.len() < 256 {
            colors.push(palette[colors.len() % palette.len()]);
        }
        colors
    }

    /// Full recompute of the smoothed buffer from raw. Called only when tau changes.
    fn recompute_smoothing(&mut self) {
        let raw_lock = self.raw_data.lock().unwrap();
        let mut smooth_lock = self.smoothed_data.lock().unwrap();
        for (i, raw_series) in raw_lock.iter().enumerate() {
            let smooth_series = &mut smooth_lock[i];
            smooth_series.data.clear();
            if let Some(first) = raw_series.first() {
                smooth_series.push(*first);
                let (mut prev_y, mut prev_t) = (first[1], first[0]);
                for point in raw_series.data.iter().skip(1) {
                    let (cur_t, cur_x) = (point[0], point[1]);
                    let y = if self.tau <= 1e-6 {
                        cur_x
                    } else {
                        let alpha = 1.0 - (-(cur_t - prev_t).max(0.0) / self.tau).exp();
                        alpha * cur_x + (1.0 - alpha) * prev_y
                    };
                    smooth_series.push([cur_t, y]);
                    prev_y = y;
                    prev_t = cur_t;
                }
            }
        }
    }
}

fn format_x_val(x: f64, is_ts: bool) -> String {
    if is_ts {
        if let Some(dt) = DateTime::from_timestamp(x as i64, ((x % 1.0) * 1e9) as u32) {
            return dt.with_timezone(&Local).format("%H:%M:%S").to_string();
        }
    }
    format!("{:.0}", x)
}

impl eframe::App for LivePlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- 1. SETTINGS PANEL ---
        let panel_width = if self.side_panel_collapsed {
            28.0
        } else {
            let font_id = TextStyle::Button.resolve(&ctx.style());
            let max_label_w = ctx.fonts(|f| {
                self.labels
                    .iter()
                    .map(|l| {
                        f.layout_no_wrap(l.clone(), font_id.clone(), Color32::WHITE)
                            .rect
                            .width()
                    })
                    .fold(0.0f32, f32::max)
            });
            (max_label_w + 90.0).max(220.0).min(600.0)
        };

        egui::SidePanel::right("settings_panel")
            .resizable(false)
            .exact_width(panel_width)
            .show(ctx, |ui| {
                if self.side_panel_collapsed {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        if ui.button("⏴").clicked() {
                            self.side_panel_collapsed = false;
                        }
                    });
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("⏵").clicked() {
                            self.side_panel_collapsed = true;
                        }
                        ui.heading("Settings");
                    });
                    ui.separator();
                    ui.label(egui::RichText::new("Exponential Smoothing").strong());
                    let mut tau_changed = false;
                    ui.horizontal_wrapped(|ui| {
                        for &(label, val) in &[
                            ("None", 0.000001f64),
                            ("1s", 1.0),
                            ("5s", 5.0),
                            ("15s", 15.0),
                        ] {
                            let active = if val <= 1e-6 {
                                self.tau <= 1e-6
                            } else {
                                (self.tau - val).abs() < 0.01
                            };
                            if ui.selectable_label(active, label).clicked() {
                                self.tau = val;
                                tau_changed = true;
                            }
                        }
                    });
                    if ui
                        .add(
                            egui::Slider::new(&mut self.tau, 0.000001..=self.max_tau)
                                .show_value(true)
                                .text("τ (s)"),
                        )
                        .changed()
                    {
                        tau_changed = true;
                    }
                    if tau_changed {
                        *self.tau_shared.lock().unwrap() = self.tau;
                        self.recompute_smoothing();
                    }
                    ui.add_space(10.0);
                    ui.separator();
                    ui.label(egui::RichText::new("Visibility").strong());
                    ui.horizontal(|ui| {
                        if ui.button("All").clicked() {
                            self.visible.iter_mut().for_each(|v| *v = true);
                        }
                        if ui.button("None").clicked() {
                            self.visible.iter_mut().for_each(|v| *v = false);
                        }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for i in 0..self.labels.len() {
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(12.0, 12.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(rect, 2.0, self.colors[i]);
                                if ui
                                    .selectable_label(self.visible[i], &self.labels[i])
                                    .clicked()
                                {
                                    self.visible[i] = !self.visible[i];
                                }
                            });
                        }
                    });
                }
            });

        // --- 2. CENTRAL PANEL ---
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut first_x = 0.0f64;
            let mut last_x = 0.0f64;
            let mut global_min_y = f64::INFINITY;
            let mut global_max_y = f64::NEG_INFINITY;

            {
                let d = self.raw_data.lock().unwrap();
                let mut found_x = false;
                for buf in d.iter() {
                    if let (Some(f), Some(l)) = (buf.first(), buf.last()) {
                        if !found_x || f[0] < first_x {
                            first_x = f[0];
                        }
                        if !found_x || l[0] > last_x {
                            last_x = l[0];
                        }
                        found_x = true;
                    }
                    for p in buf.data.iter() {
                        global_min_y = global_min_y.min(p[1]);
                        global_max_y = global_max_y.max(p[1]);
                    }
                }
            }
            if global_min_y.is_infinite() {
                global_min_y = -1.0;
                global_max_y = 1.0;
            }

            let mut trigger = false;
            let layer_id = ui.layer_id();

            // HEADER
            ui.horizontal(|ui| {
                ui.heading(&self.title);

                let font_id = TextStyle::Body.resolve(ui.style());
                let max_status_w = ui.fonts(|f| {
                    ["• LIVE", "EXPLORE", "⚠️ Ended"]
                        .iter()
                        .map(|s| {
                            f.layout_no_wrap(s.to_string(), font_id.clone(), Color32::WHITE)
                                .rect
                                .width()
                        })
                        .fold(0.0f32, f32::max)
                });

                ui.add_sized([max_status_w + 10.0, 20.0], |ui: &mut egui::Ui| {
                    ui.centered_and_justified(|ui| {
                        if self.stream_ended.load(Ordering::Relaxed) {
                            ui.colored_label(Color32::GOLD, "⚠️ Ended")
                        } else if self.auto_follow {
                            ui.colored_label(Color32::from_rgb(100, 200, 255), "• LIVE")
                        } else {
                            ui.weak("EXPLORE")
                        }
                    })
                    .response
                });

                ui.separator();
                ui.label("X-Anchor:");
                ui.selectable_value(&mut self.anchor_x, XAnchor::Left, "L");
                ui.selectable_value(&mut self.anchor_x, XAnchor::Center, "C");
                ui.selectable_value(&mut self.anchor_x, XAnchor::Right, "R");

                ui.separator();
                ui.label("X-Zoom:");
                let cur_x_w = self.default_width / self.x_zoom;
                let cur_x_anchor = match self.anchor_x {
                    XAnchor::Left => self.scroll_offset,
                    XAnchor::Center => self.scroll_offset + (cur_x_w / 2.0),
                    XAnchor::Right => self.scroll_offset + cur_x_w,
                };

                let slider_w = (ui.available_width() - 115.0).max(50.0);
                let min_zx =
                    (self.default_width / (last_x - first_x).max(self.default_width)).max(0.0001);
                let zx_resp = ui.add_sized(
                    [slider_w, 20.0],
                    egui::Slider::new(&mut self.x_zoom, min_zx..=2000.0)
                        .show_value(false)
                        .logarithmic(true),
                );
                if zx_resp.changed() {
                    self.auto_follow = false;
                    trigger = true;
                    let new_x_w = self.default_width / self.x_zoom;
                    self.scroll_offset = match self.anchor_x {
                        XAnchor::Left => cur_x_anchor,
                        XAnchor::Center => cur_x_anchor - (new_x_w / 2.0),
                        XAnchor::Right => cur_x_anchor - new_x_w,
                    };
                }
                if ui.button("Reset Viewport").clicked() {
                    self.x_zoom = 1.0;
                    self.y_zoom = 1.0;
                    self.auto_follow = true;
                    trigger = true;
                }
            });

            if self.auto_follow {
                self.scroll_offset = (last_x - (self.default_width / self.x_zoom)).max(first_x);
            }

            let body_layout =
                egui::Layout::left_to_right(egui::Align::Min).with_cross_justify(true);
            ui.with_layout(body_layout, |ui| {
                let cur_y_h = (self.y_max - self.y_min).max(0.0001);
                let cur_y_anchor = match self.anchor_y {
                    YAnchor::Top => self.y_max,
                    YAnchor::Center => self.y_min + (cur_y_h / 2.0),
                    YAnchor::Bottom => self.y_min,
                };

                ui.vertical(|ui| {
                    ui.label("Y-Anchor");
                    ui.selectable_value(&mut self.anchor_y, YAnchor::Top, "T");
                    ui.selectable_value(&mut self.anchor_y, YAnchor::Center, "C");
                    ui.selectable_value(&mut self.anchor_y, YAnchor::Bottom, "B");
                    ui.add_space(10.0);

                    let min_zy = (self.y_nat_h / (global_max_y - global_min_y).max(0.1)).min(1.0);
                    if ui
                        .add_sized(
                            [20.0, ui.available_height()],
                            egui::Slider::new(&mut self.y_zoom, min_zy..=50.0)
                                .vertical()
                                .show_value(false)
                                .logarithmic(true),
                        )
                        .changed()
                    {
                        self.auto_follow = false;
                        trigger = true;
                        let new_y_h = self.y_nat_h / self.y_zoom;
                        match self.anchor_y {
                            YAnchor::Top => {
                                self.y_max = cur_y_anchor;
                                self.y_min = self.y_max - new_y_h;
                            }
                            YAnchor::Center => {
                                self.y_min = cur_y_anchor - (new_y_h / 2.0);
                                self.y_max = cur_y_anchor + (new_y_h / 2.0);
                            }
                            YAnchor::Bottom => {
                                self.y_min = cur_y_anchor;
                                self.y_max = self.y_min + new_y_h;
                            }
                        }
                    }
                });

                // The pixel width of the plot area is not known until after layout, so we
                // use available_width minus the Y-controls column (~24px) as a proxy.
                let plot_pixel_w = (ui.available_width() - 24.0).max(400.0) as usize;

                let mut plot = Plot::new("lp")
                    .height(ui.available_height() - 4.0)
                    .width(ui.available_width())
                    .auto_bounds([false, false].into())
                    .allow_zoom(true)
                    .allow_drag(true)
                    .show_x(false)
                    .show_y(false)
                    .label_formatter(|_, _| String::new());

                if self.is_ts_mode {
                    plot = plot.x_grid_spacer(move |input: GridInput| {
                        let span = input.bounds.1 - input.bounds.0;
                        let steps = [
                            86400.0, 43200.0, 21600.0, 14400.0, 10800.0, 7200.0, 3600.0, 1800.0,
                            900.0, 600.0, 300.0, 120.0, 60.0, 30.0, 15.0, 10.0, 5.0, 1.0,
                        ];
                        let major_step = steps
                            .iter()
                            .copied()
                            .find(|&s| span / s >= 4.0)
                            .unwrap_or(1.0);
                        let minor_step = major_step / 4.0;
                        let sample_ts = input.bounds.0 as i64;
                        let local_offset = if let Some(dt) = DateTime::from_timestamp(sample_ts, 0)
                        {
                            dt.with_timezone(&Local).offset().fix().local_minus_utc() as f64
                        } else {
                            0.0
                        };
                        let mut marks = Vec::new();
                        let start_aligned = ((input.bounds.0 + local_offset) / minor_step).ceil()
                            * minor_step
                            - local_offset;
                        let mut val = start_aligned;
                        while val <= input.bounds.1 {
                            marks.push(GridMark {
                                value: val,
                                step_size: major_step,
                            });
                            val += minor_step;
                        }
                        marks
                    });

                    plot = plot.x_axis_formatter(move |mark, range| {
                        let local_offset =
                            if let Some(dt) = DateTime::from_timestamp(mark.value as i64, 0) {
                                dt.with_timezone(&Local).offset().fix().local_minus_utc() as f64
                            } else {
                                0.0
                            };
                        if ((mark.value + local_offset) % mark.step_size).abs() < 1e-5 {
                            if let Some(dt) = DateTime::from_timestamp(mark.value as i64, 0) {
                                let ldt = dt.with_timezone(&Local);
                                let span = *range.end() - *range.start();
                                if span > 86400.0 {
                                    ldt.format("%m/%d %H:%M").to_string()
                                } else if mark.step_size >= 60.0 {
                                    ldt.format("%H:%M").to_string()
                                } else {
                                    ldt.format("%H:%M:%S").to_string()
                                }
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        }
                    });
                } else {
                    plot = plot.x_axis_formatter(move |m, _| format!("{:.0}", m.value));
                }

                if let Some(pos) = self.legend_corner {
                    plot = plot.legend(Legend::default().position(pos));
                }
                for &y in &self.include_y_values {
                    plot = plot.include_y(y);
                }

                let data_arc = self.smoothed_data.clone();
                let labels = self.labels.clone();
                let colors = self.colors.clone();
                let is_ts = self.is_ts_mode;
                let include_y = self.include_y_values.clone();
                let visible = self.visible.clone();
                let def_w = self.default_width;
                let max_render = self.max_render_points;

                let plot_res = plot.show(ui, |plot_ui| {
                    if plot_ui.pointer_coordinate_drag_delta().length() > 0.0
                        || plot_ui.ctx().input(|i| i.raw_scroll_delta.y).abs() > 0.0
                    {
                        self.auto_follow = false;
                    }
                    let b = plot_ui.plot_bounds();

                    if self.auto_follow {
                        let width = def_w / self.x_zoom;
                        let x_start = last_x - width;
                        let mut min_y_vis = f64::INFINITY;
                        let mut max_y_vis = f64::NEG_INFINITY;
                        let d = data_arc.lock().unwrap();
                        for (i, buf) in d.iter().enumerate() {
                            if !visible[i] {
                                continue;
                            }
                            // Use binary search to find the viewport start index.
                            let si = buf.search_x(x_start).saturating_sub(1);
                            for p in buf.data.iter().skip(si) {
                                min_y_vis = min_y_vis.min(p[1]);
                                max_y_vis = max_y_vis.max(p[1]);
                            }
                        }
                        for &y in &include_y {
                            min_y_vis = min_y_vis.min(y);
                            max_y_vis = max_y_vis.max(y);
                        }
                        let base = if min_y_vis.is_infinite() {
                            (-1.0, 1.0)
                        } else {
                            let p = (max_y_vis - min_y_vis).max(0.1) * 0.05;
                            (min_y_vis - p, max_y_vis + p)
                        };
                        self.y_min = base.0;
                        self.y_max = base.1;
                        self.y_nat_h = (base.1 - base.0).max(0.001);
                        self.y_zoom = 1.0;
                        self.scroll_offset = x_start;
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [x_start, base.0],
                            [last_x, base.1],
                        ));
                    } else if trigger {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [self.scroll_offset, self.y_min],
                            [self.scroll_offset + (def_w / self.x_zoom), self.y_max],
                        ));
                    } else {
                        self.scroll_offset = b.min()[0];
                        self.y_min = b.min()[1];
                        self.y_max = b.max()[1];
                        if b.width() > 0.0 {
                            self.x_zoom = def_w / b.width();
                        }
                        if b.height() > 0.0 {
                            self.y_zoom = (self.y_nat_h / b.height()).max(0.001);
                        }
                    }

                    // --- LTTB decimation render ---
                    let d = data_arc.lock().unwrap();
                    // Target: 4 samples per pixel (2x Nyquist), capped at max_render_points.
                    let render_target = max_render.unwrap_or(plot_pixel_w * 4).max(64);

                    for (i, buf) in d.iter().enumerate() {
                        if !visible[i] || buf.is_empty() {
                            continue;
                        }
                        let start_idx = buf.search_x(b.min()[0]).saturating_sub(1);
                        let end_idx = buf.search_x(b.max()[0]).min(buf.len());
                        let count = end_idx.saturating_sub(start_idx);
                        if count == 0 {
                            continue;
                        }

                        let pts = if count <= render_target {
                            // No decimation needed — pass points directly.
                            PlotPoints::new((start_idx..end_idx).map(|j| buf.data[j]).collect())
                        } else {
                            let decimated = lttb(buf, start_idx, end_idx, render_target);
                            PlotPoints::new(decimated.into_iter().collect())
                        };
                        plot_ui.line(Line::new(pts).name(&labels[i]).color(colors[i]));
                    }

                    // --- Tooltip: search each series independently ---
                    if let Some(mp) = plot_ui.pointer_coordinate() {
                        let bounds = plot_ui.plot_bounds();
                        // Normalize Y distances by aspect ratio so pixels are equal.
                        let y_scale = if bounds.height() > 0.0 {
                            bounds.width() / bounds.height()
                        } else {
                            1.0
                        };
                        let snap_sq = (bounds.width() * 0.015).powi(2);
                        let mut best: Option<(usize, f64, f64)> = None;
                        let mut best_dsq = f64::INFINITY;

                        for si in 0..labels.len() {
                            if !visible[si] {
                                continue;
                            }
                            if let Some(buf) = d.get(si) {
                                if buf.is_empty() {
                                    continue;
                                }
                                let idx = buf.search_x(mp.x);
                                for j in idx.saturating_sub(1)..(idx + 2).min(buf.len()) {
                                    let p = buf.data[j];
                                    let dx = p[0] - mp.x;
                                    let dy = (p[1] - mp.y) * y_scale;
                                    let dsq = dx * dx + dy * dy;
                                    if dsq < best_dsq {
                                        best_dsq = dsq;
                                        best = Some((si, p[0], p[1]));
                                    }
                                }
                            }
                        }
                        if best_dsq < snap_sq {
                            if let Some((si, x, y)) = best {
                                let xf = format_x_val(x, is_ts);
                                egui::show_tooltip_at_pointer(
                                    plot_ui.ctx(),
                                    layer_id,
                                    egui::Id::new("tt"),
                                    |ui: &mut egui::Ui| {
                                        ui.label(format!("{}: {:.4}\nX: {}", labels[si], y, xf));
                                    },
                                );
                            }
                        }
                    }

                    // Event-driven repaint: request another frame only while the view
                    // is still changing (animation/drag in progress). Once it settles,
                    // we stop requesting and go fully idle until the next external event
                    // (new data from reader thread, user input, or egui hover/tooltip).
                    let cur_v = [b.min()[0], b.min()[1], b.max()[0], b.max()[1]];
                    if self.last_view_rect != Some(cur_v) {
                        self.last_view_rect = Some(cur_v);
                        self.view_settling = true;
                        ctx.request_repaint();
                    } else if self.view_settling {
                        // View has stabilised — one final repaint to render the settled
                        // state, then go idle.
                        self.view_settling = false;
                        ctx.request_repaint();
                    }
                });
                if trigger || plot_res.response.dragged() || plot_res.response.double_clicked() {
                    ui.ctx().request_repaint();
                }
            });
        });
    }
}

// ---------------------------------------------------------------------------
// Input reader: spawns a background thread to read from stdin, a file, or a
// Unix socket.  Returns the channel sender used to push [x, y_0..y_n] rows.
// ---------------------------------------------------------------------------

/// Parsed row from any input mode.
struct DataRow {
    x: f64,
    /// Values indexed by display order.
    values: Vec<(usize, f64)>, // (display_index, value)
}

fn spawn_reader(
    args: Args,
    raw_buffer: Arc<Mutex<Vec<SeriesBuffer>>>,
    smooth_buffer: Arc<Mutex<Vec<SeriesBuffer>>>,
    tau_shared: Arc<Mutex<f64>>,
    stream_ended: Arc<AtomicBool>,
    input_to_display_map: Vec<usize>,
    display_labels: Vec<String>,
    label_count: usize,
    ctx: egui::Context,
    csv_out: Option<Arc<Mutex<Box<dyn io::Write + Send>>>>,
) {
    let is_ts = args.timestamp || args.json;
    let period = args.sample_period;
    let input_path = args.input.clone();
    let use_json = args.json;
    let expected = if is_ts { label_count + 1 } else { label_count };

    thread::spawn(move || {
        let reader: Box<dyn BufRead + Send> = if input_path == "-" {
            Box::new(io::BufReader::new(io::stdin()))
        } else {
            match std::fs::File::open(&input_path) {
                Ok(f) => Box::new(io::BufReader::new(f)),
                Err(e) => {
                    eprintln!("FATAL ERROR: cannot open '{}': {}", input_path, e);
                    std::process::exit(1);
                }
            }
        };

        let mut seq = 0u64;
        for line in reader.lines() {
            let line_str = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let trimmed = line_str.trim();
            if trimmed.is_empty() {
                continue;
            }

            let row: Option<DataRow> = if use_json {
                parse_json_row(trimmed, &display_labels)
            } else {
                parse_csv_row(trimmed, expected, is_ts, seq, period, &input_to_display_map)
            };

            let row = match row {
                Some(r) => r,
                None => continue,
            };

            {
                let mut rb = raw_buffer.lock().unwrap();
                let mut sb = smooth_buffer.lock().unwrap();
                let t = *tau_shared.lock().unwrap();

                for (display_i, v) in &row.values {
                    let display_i = *display_i;
                    if display_i >= rb.len() {
                        continue;
                    }
                    let x = row.x;

                    rb[display_i].push([x, *v]);
                    // Keep sorted if timestamps can arrive out of order
                    if is_ts {
                        let len = rb[display_i].len();
                        let mut j = len - 1;
                        while j > 0 && rb[display_i].data[j][0] < rb[display_i].data[j - 1][0] {
                            rb[display_i].data.swap(j, j - 1);
                            j -= 1;
                        }
                    }

                    let y = if let Some(last) = sb[display_i].last() {
                        let dt = (x - last[0]).max(0.0);
                        if t <= 1e-6 {
                            *v
                        } else {
                            let alpha = 1.0 - (-dt / t).exp();
                            alpha * v + (1.0 - alpha) * last[1]
                        }
                    } else {
                        *v
                    };
                    sb[display_i].push([x, y]);
                }
                seq += 1;

                if let Some(ref out_arc) = csv_out {
                    // Write CSV line: x, v0, v1, ...
                    let mut out = out_arc.lock().unwrap();
                    let _ = write!(out, "{}", row.x);
                    // Rebuild full value list in display order
                    let mut vals = vec![f64::NAN; label_count];
                    for (di, v) in &row.values {
                        if *di < label_count {
                            vals[*di] = *v;
                        }
                    }
                    for v in &vals {
                        if v.is_nan() {
                            let _ = write!(out, ",");
                        } else {
                            let _ = write!(out, ",{}", v);
                        }
                    }
                    let _ = writeln!(out);
                }
            }
            ctx.request_repaint();
        }
        stream_ended.store(true, Ordering::Relaxed);
        ctx.request_repaint();
    });
}

fn parse_csv_row(
    line: &str,
    expected: usize,
    is_ts: bool,
    seq: u64,
    period: f64,
    map: &[usize],
) -> Option<DataRow> {
    let tokens: Vec<&str> = line
        .split(|c| c == ',' || c == ' ')
        .filter(|s| !s.is_empty())
        .collect();
    if tokens.len() != expected {
        eprintln!(
            "WARNING: expected {} cols, found {} — skipping line: {}",
            expected,
            tokens.len(),
            line
        );
        return None;
    }
    let (x, data_tokens) = if is_ts {
        match tokens[0].parse::<f64>() {
            Ok(v) => (v, &tokens[1..]),
            Err(_) => {
                eprintln!("WARNING: cannot parse timestamp '{}' — skipping", tokens[0]);
                return None;
            }
        }
    } else {
        (seq as f64 * period, &tokens[..])
    };

    let mut values = Vec::new();
    for (orig_i, token) in data_tokens.iter().enumerate() {
        let low = token.to_ascii_lowercase();
        if low == "none" || low == "null" || low == "nan" {
            continue;
        }
        if let Ok(v) = token.parse::<f64>() {
            values.push((map[orig_i], v));
        }
    }
    Some(DataRow { x, values })
}

/// Parse a newline-delimited JSON row of the form:
///   {"t": 1234567890.0, "values": {"label_a": 1.2, "label_b": 3.4}}
///
/// `display_labels` is the ordered list of label strings shown in the UI.
/// We do a linear search per key — fine for ≤ a few hundred labels.
fn parse_json_row(line: &str, display_labels: &[String]) -> Option<DataRow> {
    let t = extract_json_f64(line, "\"t\"")?;
    let values_start = line.find("\"values\"")?;
    let obj_start = line[values_start..].find('{')? + values_start + 1;
    let obj_end = line[obj_start..].find('}')? + obj_start;
    let obj = &line[obj_start..obj_end];

    let mut values = Vec::new();
    for pair in obj.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let colon = match pair.find(':') {
            Some(c) => c,
            None => continue,
        };
        let key = pair[..colon].trim().trim_matches('"');
        let val_str = pair[colon + 1..].trim();
        if let Ok(v) = val_str.parse::<f64>() {
            if let Some(display_i) = display_labels.iter().position(|l| l == key) {
                values.push((display_i, v));
            } else {
                eprintln!("WARNING: JSON key '{}' not in label list — ignoring", key);
            }
        }
    }
    Some(DataRow { x: t, values })
}

fn extract_json_f64(s: &str, key: &str) -> Option<f64> {
    let ki = s.find(key)?;
    let after = &s[ki + key.len()..];
    let ci = after.find(':')?;
    let num_str = after[ci + 1..].trim_start();
    let end = num_str
        .find(|c: char| {
            !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+'
        })
        .unwrap_or(num_str.len());
    num_str[..end].parse().ok()
}

// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    let _is_ts = args.timestamp || args.json;

    let raw_labels = args
        .labels
        .clone()
        .unwrap_or_else(|| vec!["Series 1".to_string()]);
    let label_count = raw_labels.len();

    let mut map: Vec<(usize, String)> = raw_labels.into_iter().enumerate().collect();
    if args.sort_labels {
        map.sort_by(|a, b| a.1.cmp(&b.1));
    }
    let display_labels: Vec<String> = map.iter().map(|m| m.1.clone()).collect();
    let input_to_display_map: Vec<usize> = {
        let mut inv = vec![0usize; label_count];
        for (display_idx, (original_idx, _)) in map.iter().enumerate() {
            inv[*original_idx] = display_idx;
        }
        inv
    };

    let max_pts = args.max_points;
    let raw_buffer: Arc<Mutex<Vec<SeriesBuffer>>> = Arc::new(Mutex::new(
        (0..label_count)
            .map(|_| SeriesBuffer::new(max_pts))
            .collect(),
    ));
    let smooth_buffer: Arc<Mutex<Vec<SeriesBuffer>>> = Arc::new(Mutex::new(
        (0..label_count)
            .map(|_| SeriesBuffer::new(max_pts))
            .collect(),
    ));

    let tau_shared: Arc<Mutex<f64>> = Arc::new(Mutex::new(0.000001));
    let stream_ended = Arc::new(AtomicBool::new(false));

    let csv_out: Option<Arc<Mutex<Box<dyn io::Write + Send>>>> = args.output.as_ref().map(|path| {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| {
                eprintln!("FATAL ERROR: cannot open output '{}': {}", path, e);
                std::process::exit(1);
            });
        // Write CSV header if file is empty.
        let mut w: Box<dyn io::Write + Send> = Box::new(io::BufWriter::new(file));
        let header = format!("t,{}\n", display_labels.join(","));
        let _ = w.write_all(header.as_bytes());
        Arc::new(Mutex::new(w))
    });

    let app_args = args.clone();

    eframe::run_native(
        "Live Plotter",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 850.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            spawn_reader(
                app_args.clone(),
                Arc::clone(&raw_buffer),
                Arc::clone(&smooth_buffer),
                Arc::clone(&tau_shared),
                Arc::clone(&stream_ended),
                input_to_display_map,
                display_labels.clone(),
                label_count,
                ctx,
                csv_out,
            );
            Ok(Box::new(LivePlotApp::new(
                app_args,
                raw_buffer,
                smooth_buffer,
                tau_shared,
                stream_ended,
                display_labels,
            )))
        }),
    )
    .unwrap();
}
