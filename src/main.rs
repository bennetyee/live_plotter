use chrono::{DateTime, Local, Offset};
use clap::Parser;
use eframe::egui;
use egui::{Color32, TextStyle};
use egui_plot::{Corner, GridInput, GridMark, Legend, Line, Plot, PlotBounds, PlotPoints};
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "High-Performance Live Multi-Series Plotter")]
struct Args {
    /// Maximum number of data points to display in the total buffer per series
    #[arg(short, long, default_value_t = 1000000)]
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

    /// The title displayed at the top of the graph (Long form only)
    #[arg(long, default_value = "Live Time-Series Feed")]
    title: String,

    /// Legend position: LeftTop, RightTop, LeftBottom, RightBottom, or None
    #[arg(long, default_value = "LeftTop")]
    legend_pos: String,

    /// Maximum smoothing time constant (tau) in seconds
    #[arg(long, default_value_t = 60.0)]
    max_tau: f64,
}

type SeriesBuffer = Vec<[f64; 2]>;

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
            is_ts_mode: args.timestamp,
            side_panel_collapsed: false,
            tau: 0.000001,
            max_tau: args.max_tau,
            default_width: args.viewport_width,
            x_zoom: 1.0,
            scroll_offset: 0.0,
            y_zoom: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            y_nat_h: 1.0,
            auto_follow: true,
            anchor_x: XAnchor::Center,
            anchor_y: YAnchor::Center,
            last_view_rect: None,
        }
    }

    fn generate_palette(user_colors: Option<Vec<String>>) -> Vec<Color32> {
        let palette = vec![
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
        let mut colors = Vec::new();
        if let Some(h) = user_colors {
            for hex in h {
                if let Ok(c) = Color32::from_hex(&hex) {
                    colors.push(c);
                }
            }
        }
        while colors.len() < 100 {
            colors.push(palette[colors.len() % palette.len()]);
        }
        colors
    }

    fn recompute_smoothing(&mut self) {
        let raw_lock = self.raw_data.lock().unwrap();
        let mut smooth_lock = self.smoothed_data.lock().unwrap();
        for (i, raw_series) in raw_lock.iter().enumerate() {
            let smooth_series = &mut smooth_lock[i];
            smooth_series.clear();
            if let Some(first) = raw_series.first() {
                smooth_series.push(*first);
                let (mut prev_y, mut prev_t) = (first[1], first[0]);
                for point in raw_series.iter().skip(1) {
                    let (cur_t, cur_x) = (point[0], point[1]);
                    let alpha = 1.0 - (-(cur_t - prev_t).max(0.0) / self.tau).exp();
                    let y = if self.tau <= 1e-6 {
                        cur_x
                    } else {
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
                    .fold(0.0, f32::max)
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
                        if ui.selectable_label(self.tau <= 1e-6, "None").clicked() {
                            self.tau = 0.000001;
                            tau_changed = true;
                        }
                        if ui
                            .selectable_label((self.tau - 1.0).abs() < 0.01, "1s")
                            .clicked()
                        {
                            self.tau = 1.0;
                            tau_changed = true;
                        }
                        if ui
                            .selectable_label((self.tau - 5.0).abs() < 0.01, "5s")
                            .clicked()
                        {
                            self.tau = 5.0;
                            tau_changed = true;
                        }
                        if ui
                            .selectable_label((self.tau - 15.0).abs() < 0.01, "15s")
                            .clicked()
                        {
                            self.tau = 15.0;
                            tau_changed = true;
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
                            for v in self.visible.iter_mut() {
                                *v = true;
                            }
                        }
                        if ui.button("None").clicked() {
                            for v in self.visible.iter_mut() {
                                *v = false;
                            }
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
            let mut first_x = 0.0;
            let mut last_x = 0.0;
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
                    // For vertical zoom range logic
                    for p in buf.iter() {
                        if p[1] < global_min_y {
                            global_min_y = p[1];
                        }
                        if p[1] > global_max_y {
                            global_max_y = p[1];
                        }
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
                    let w1 = f
                        .layout_no_wrap("• LIVE".into(), font_id.clone(), Color32::WHITE)
                        .rect
                        .width();
                    let w2 = f
                        .layout_no_wrap("EXPLORE".into(), font_id.clone(), Color32::WHITE)
                        .rect
                        .width();
                    let w3 = f
                        .layout_no_wrap("⚠️ Ended".into(), font_id.clone(), Color32::WHITE)
                        .rect
                        .width();
                    w1.max(w2).max(w3)
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

                // Vertical column for Y-controls
                ui.vertical(|ui| {
                    ui.label("Y-Anchor");
                    ui.selectable_value(&mut self.anchor_y, YAnchor::Top, "T");
                    ui.selectable_value(&mut self.anchor_y, YAnchor::Center, "C");
                    ui.selectable_value(&mut self.anchor_y, YAnchor::Bottom, "B");
                    ui.add_space(10.0);

                    // Dynamic Y-zoom range calculation
                    // We ensure that we can always zoom out to see the entire buffer's range
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
                                "".into()
                            }
                        } else {
                            "".into()
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
                            for p in buf.iter().filter(|p| p[0] >= x_start) {
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

                    // Optimized decimation render
                    let d = data_arc.lock().unwrap();
                    for (i, buf) in d.iter().enumerate() {
                        if !visible[i] || buf.is_empty() {
                            continue;
                        }
                        let start_idx = buf
                            .binary_search_by(|p| p[0].partial_cmp(&b.min()[0]).unwrap())
                            .unwrap_or_else(|e| e)
                            .saturating_sub(1);
                        let end_idx = buf
                            .binary_search_by(|p| p[0].partial_cmp(&b.max()[0]).unwrap())
                            .unwrap_or_else(|e| e)
                            .min(buf.len());
                        let count = end_idx - start_idx;
                        if count == 0 {
                            continue;
                        }

                        let step = (count / 2000).max(1);
                        let pts = PlotPoints::from_parametric_callback(
                            move |t| {
                                let bin_idx = t.round() as usize / 2;
                                let is_max = t.round() as usize % 2 == 1;
                                let b_start = start_idx + (bin_idx * step);
                                let b_end = (b_start + step).min(end_idx);
                                if b_start >= buf.len() {
                                    return (0.0, 0.0);
                                }
                                let mut l_min = f64::INFINITY;
                                let mut l_max = f64::NEG_INFINITY;
                                let (mut mi, mut mai) = (b_start, b_start);
                                for j in b_start..b_end {
                                    if buf[j][1] < l_min {
                                        l_min = buf[j][1];
                                        mi = j;
                                    }
                                    if buf[j][1] > l_max {
                                        l_max = buf[j][1];
                                        mai = j;
                                    }
                                }
                                let (f, s) = if mi < mai { (mi, mai) } else { (mai, mi) };
                                let p = if is_max { buf[s] } else { buf[f] };
                                (p[0], p[1])
                            },
                            0.0..=((count / step * 2) as f64),
                            (count / step * 2) + 1,
                        );
                        plot_ui.line(Line::new(pts).name(&labels[i]).color(colors[i]));
                    }

                    if let Some(mp) = plot_ui.pointer_coordinate() {
                        if let Some(rb) = d.get(0) {
                            let idx = rb
                                .binary_search_by(|p| p[0].partial_cmp(&mp.x).unwrap())
                                .unwrap_or_else(|e| e);
                            let mut best = None;
                            let mut bd = f64::INFINITY;
                            for i in (idx.saturating_sub(1))..(idx + 1).min(rb.len()) {
                                for si in 0..labels.len() {
                                    if !visible[si] {
                                        continue;
                                    }
                                    if let Some(v) = d.get(si).and_then(|b| b.get(i)) {
                                        let dx = v[0] - mp.x;
                                        let dy = (v[1] - mp.y)
                                            * (plot_ui.plot_bounds().width()
                                                / plot_ui.plot_bounds().height().max(0.1));
                                        let dsq = dx * dx + dy * dy;
                                        if dsq < bd {
                                            bd = dsq;
                                            best = Some((si, v[0], v[1]));
                                        }
                                    }
                                }
                            }
                            if let Some((si, x, y)) = best {
                                if bd < (plot_ui.plot_bounds().width() * 0.015).powi(2) {
                                    let xf = format_x_val(x, is_ts);
                                    let l = labels[si].clone();
                                    egui::show_tooltip_at_pointer(
                                        plot_ui.ctx(),
                                        layer_id,
                                        egui::Id::new("tt"),
                                        |ui: &mut egui::Ui| {
                                            ui.label(format!("{}: {:.4}\nX: {}", l, y, xf));
                                        },
                                    );
                                }
                            }
                        }
                    }

                    let cur_v = [b.min()[0], b.min()[1], b.max()[0], b.max()[1]];
                    if self.last_view_rect != Some(cur_v) {
                        ctx.request_repaint();
                        self.last_view_rect = Some(cur_v);
                    }
                });
                if trigger || plot_res.response.dragged() {
                    ui.ctx().request_repaint();
                }
            });
        });
    }
}

fn main() {
    let args = Args::parse();
    let is_ts = args.timestamp;
    let period = args.sample_period;
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
        let mut inv = vec![0; label_count];
        for (display_idx, (original_idx, _)) in map.iter().enumerate() {
            inv[*original_idx] = display_idx;
        }
        inv
    };
    let expected = if is_ts { label_count + 1 } else { label_count };
    let raw_buffer: Arc<Mutex<Vec<SeriesBuffer>>> =
        Arc::new(Mutex::new(vec![Vec::new(); label_count]));
    let smooth_buffer: Arc<Mutex<Vec<SeriesBuffer>>> =
        Arc::new(Mutex::new(vec![Vec::new(); label_count]));
    let raw_thread = Arc::clone(&raw_buffer);
    let smooth_thread = Arc::clone(&smooth_buffer);
    let tau_shared = Arc::new(Mutex::new(0.000001));
    let tau_thread = Arc::clone(&tau_shared);
    let stream_ended = Arc::new(AtomicBool::new(false));
    let stream_ended_thread = Arc::clone(&stream_ended);
    let max_pts = args.max_points;
    let app_args = args.clone();
    eframe::run_native(
        "Live Plotter",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 850.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                let mut seq = 0;
                for (_, line) in stdin.lock().lines().enumerate() {
                    if let Ok(line_str) = line {
                        let trimmed = line_str.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let tokens: Vec<&str> = trimmed
                            .split(|c| c == ',' || c == ' ')
                            .filter(|s| !s.is_empty())
                            .collect();
                        if tokens.len() != expected {
                            eprintln!(
                                "FATAL ERROR: expected {} cols, found {}.",
                                expected,
                                tokens.len()
                            );
                            std::process::exit(1);
                        }
                        let (x, data_tokens) = if is_ts {
                            (
                                tokens[0].parse::<f64>().expect("Time must be float"),
                                &tokens[1..],
                            )
                        } else {
                            (seq as f64 * period, &tokens[..])
                        };
                        {
                            let mut rb = raw_thread.lock().unwrap();
                            let mut sb = smooth_thread.lock().unwrap();
                            let t = *tau_thread.lock().unwrap();
                            for (original_i, token) in data_tokens.iter().enumerate() {
                                let display_i = input_to_display_map[original_i];
                                if token.to_lowercase() == "none" || token.to_lowercase() == "null"
                                {
                                    continue;
                                }
                                if let Ok(v) = token.parse::<f64>() {
                                    if rb[display_i].len() >= max_pts {
                                        rb[display_i].remove(0);
                                    }
                                    rb[display_i].push([x, v]);
                                    if is_ts {
                                        let mut j = rb[display_i].len() - 1;
                                        while j > 0 && rb[display_i][j][0] < rb[display_i][j - 1][0]
                                        {
                                            rb[display_i].swap(j, j - 1);
                                            j -= 1;
                                        }
                                    }
                                    if sb[display_i].len() >= max_pts {
                                        sb[display_i].remove(0);
                                    }
                                    let y = if let Some(last) = sb[display_i].last() {
                                        let dt = (x - last[0]).max(0.0);
                                        if t <= 1e-6 {
                                            v
                                        } else {
                                            let alpha = 1.0 - (-dt / t).exp();
                                            alpha * v + (1.0 - alpha) * last[1]
                                        }
                                    } else {
                                        v
                                    };
                                    sb[display_i].push([x, y]);
                                }
                            }
                            seq += 1;
                        }
                        ctx.request_repaint();
                    } else {
                        break;
                    }
                }
                stream_ended_thread.store(true, Ordering::Relaxed);
                ctx.request_repaint();
            });
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
