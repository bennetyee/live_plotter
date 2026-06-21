use chrono::{DateTime, Local, Offset};
use clap::Parser;
use eframe::egui;
use egui::{Color32, TextStyle};
use egui_plot::{
    Corner, GridInput, GridMark, HLine, Line, Plot, PlotBounds, PlotPoints, Points, VLine,
};
use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "High-Performance Live Plotter")]
struct Args {
    #[arg(short, long, default_value_t = 1000000)]
    max_points: usize,
    /// Initial viewport width in X-axis units: seconds when using --timestamp, or seq*sample-period otherwise. Default 500s ≈ 8 min.
    #[arg(short, long, default_value_t = 500.0)]
    viewport_width: f64,
    #[arg(short, long, default_value_t = false)]
    timestamp: bool,
    #[arg(long, default_value_t = 5.0)]
    sample_period: f64,
    #[arg(long, num_args = 1..)]
    include_y: Vec<f64>,
    #[arg(short, long, num_args = 1..)]
    labels: Option<Vec<String>>,
    #[arg(long, default_value_t = false)]
    sort_labels: bool,
    #[arg(short, long, num_args = 1..)]
    colors: Option<Vec<String>>,
    #[arg(long, default_value = "Live Time-Series Feed")]
    title: String,
    #[arg(long, default_value_t = 60.0)]
    max_tau: f64,
    #[arg(long, default_value = "LeftTop")]
    legend_pos: String,
}

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

/// M4 min/max binning decimation, bucketed by X-axis position (time units).
///
/// Buckets are defined by pixel column in the viewport's X range, not by
/// index count. This means data gaps naturally fall into empty buckets —
/// no line is drawn across the gap, and the gap never straddles a bucket
/// boundary as the viewport moves (eliminating flicker at low zoom).
///
/// Emits up to 4 points per non-empty bucket (first, min, max, last)
/// in time order, guaranteeing every spike is visible.
/// M4 min/max binning decimation, bucketed by X-axis position (time units).
///
/// Buckets are defined by pixel column in the viewport's X range, not by
/// index count. This means data gaps naturally fall into empty buckets —
/// no line is drawn across the gap, and the gap never straddles a bucket
/// boundary as the viewport moves (eliminating the gap-flicker artifact).
///
/// Emits up to 4 points per non-empty bucket (first, min, max, last)
/// in time order, guaranteeing every spike is visible.
fn m4(
    buf: &SeriesBuffer,
    start: usize,
    end: usize,
    n_buckets: usize,
    view_min_x: f64,
    view_max_x: f64,
) -> Vec<[f64; 2]> {
    let count = end - start;
    if n_buckets == 0 || count <= n_buckets * 4 {
        return buf.data.range(start..end).copied().collect();
    }
    let mut out = Vec::with_capacity(n_buckets * 4);
    let dx = (view_max_x - view_min_x) / (n_buckets as f64).max(1.0);
    let inv_dx = if dx > 0.0 { 1.0 / dx } else { 0.0 };
    let get_bucket = |x: f64| -> i64 {
        if inv_dx == 0.0 {
            0
        } else {
            ((x - view_min_x) * inv_dx).floor() as i64
        }
    };
    let mut bs = start;
    let mut current_bucket = get_bucket(buf.data[start][0]);
    let mut min_i = start;
    let mut max_i = start;
    for i in start..end {
        let bucket = get_bucket(buf.data[i][0]);
        if bucket != current_bucket {
            if bs < i {
                let mut pts = [
                    (bs, buf.data[bs]),
                    (min_i, buf.data[min_i]),
                    (max_i, buf.data[max_i]),
                    (i - 1, buf.data[i - 1]),
                ];
                pts.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                let mut last_idx = usize::MAX;
                for (idx, p) in pts {
                    if idx != last_idx {
                        out.push(p);
                        last_idx = idx;
                    }
                }
            }
            current_bucket = bucket;
            bs = i;
            min_i = i;
            max_i = i;
        } else {
            if buf.data[i][1] < buf.data[min_i][1] {
                min_i = i;
            }
            if buf.data[i][1] > buf.data[max_i][1] {
                max_i = i;
            }
        }
    }
    if bs < end {
        let mut pts = [
            (bs, buf.data[bs]),
            (min_i, buf.data[min_i]),
            (max_i, buf.data[max_i]),
            (end - 1, buf.data[end - 1]),
        ];
        pts.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut last_idx = usize::MAX;
        for (idx, p) in pts {
            if idx != last_idx {
                out.push(p);
                last_idx = idx;
            }
        }
    }
    out
}

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
    is_ts_mode: bool,
    legend_corner: Option<Corner>,
    side_panel_collapsed: bool,
    tau: f64,
    max_tau: f64,
    default_width: f64,
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
    view_settling: bool,
    new_data: Arc<AtomicBool>,
    cached_lines: Vec<Vec<[f64; 2]>>,
    /// Cached once on first frame; width of the widest status label string.
    status_label_width: Option<f32>,
    /// Counts down after interaction ends; >0 means use coarse M4 buckets.
    interacting_frames: u8,
    /// Per-series highlight state (color swatch click in legend).
    highlighted: Vec<bool>,
    /// One-shot identify mode: next plot click finds nearest point.
    identifying: bool,
    /// Result of last identify action: (series_idx, x, y).
    identified: Option<(usize, f64, f64)>,
}

impl LivePlotApp {
    fn new(
        args: Args,
        raw_data: Arc<Mutex<Vec<SeriesBuffer>>>,
        smoothed_data: Arc<Mutex<Vec<SeriesBuffer>>>,
        tau_shared: Arc<Mutex<f64>>,
        stream_ended: Arc<AtomicBool>,
        new_data: Arc<AtomicBool>,
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
            is_ts_mode: args.timestamp,
            legend_corner,
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
            anchor_x: XAnchor::Right,
            anchor_y: YAnchor::Bottom,
            last_view_rect: None,
            view_settling: false,
            new_data,
            cached_lines: Vec::new(),
            status_label_width: None,
            interacting_frames: 0,
            highlighted: vec![false; labels.len()],
            identifying: false,
            identified: None,
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
        let mut colors = Vec::new();
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
        // --- SETTINGS PANEL ---
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
                            if ui
                                .selectable_label((self.tau - val).abs() < 0.001, label)
                                .clicked()
                            {
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
                        self.cached_lines.clear(); // force M4 recompute with new smoothed data
                    }
                    ui.add_space(10.0);
                    ui.separator();
                    ui.label(egui::RichText::new("Series").strong());
                    ui.horizontal(|ui| {
                        if ui.button("Show All").clicked() {
                            self.visible.iter_mut().for_each(|v| *v = true);
                            self.cached_lines.clear();
                        }
                        if ui.button("Hide All").clicked() {
                            self.visible.iter_mut().for_each(|v| *v = false);
                            self.cached_lines.clear();
                        }
                        if ui.button("Unhighlight").clicked() {
                            self.highlighted.iter_mut().for_each(|h| *h = false);
                            self.identified = None;
                        }
                    });
                    ui.label(
                        egui::RichText::new("● = highlight  label = show/hide")
                            .weak()
                            .small(),
                    );
                    ui.separator();
                    // Track whether the user clicked anywhere in the scroll area but
                    // NOT on a swatch or label, so we can clear highlights on blank clicks.
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        let any_highlighted = self.highlighted.iter().any(|&h| h);
                        for i in 0..self.labels.len() {
                            ui.horizontal(|ui| {
                                // Color swatch = toggle highlight
                                let swatch_color = if self.highlighted[i] {
                                    self.colors[i] // full brightness when highlighted
                                } else if any_highlighted {
                                    // Dim unselected swatches when something else is highlighted
                                    let c = self.colors[i];
                                    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 80)
                                } else {
                                    self.colors[i]
                                };
                                let (swatch_rect, swatch_resp) = ui.allocate_exact_size(
                                    egui::vec2(14.0, 14.0),
                                    egui::Sense::click(),
                                );
                                // Draw swatch with highlight ring if active
                                ui.painter().rect_filled(swatch_rect, 2.0, swatch_color);
                                if self.highlighted[i] {
                                    ui.painter().rect_stroke(
                                        swatch_rect.expand(2.0),
                                        2.0,
                                        egui::Stroke::new(2.0, Color32::WHITE),
                                    );
                                }
                                if swatch_resp.clicked() {
                                    self.highlighted[i] = !self.highlighted[i];
                                    // Clearing the last highlight also clears identify result
                                    if !self.highlighted.iter().any(|&h| h) {
                                        self.identified = None;
                                    }
                                }

                                // Label = toggle visibility
                                let label_text = if self.visible[i] {
                                    egui::RichText::new(&self.labels[i])
                                } else {
                                    egui::RichText::new(&self.labels[i]).weak()
                                };
                                if ui.selectable_label(self.visible[i], label_text).clicked() {
                                    self.visible[i] = !self.visible[i];
                                    self.cached_lines.clear();
                                }
                            });
                        }
                    });
                }
            });

        // --- CENTRAL PANEL ---
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut first_x = 0.0f64;
            let mut last_x = 0.0f64;
            let mut g_min_y = f64::INFINITY;
            let mut g_max_y = f64::NEG_INFINITY;
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
                        g_min_y = g_min_y.min(p[1]);
                        g_max_y = g_max_y.max(p[1]);
                    }
                }
            }
            if g_min_y.is_infinite() {
                g_min_y = -1.0;
                g_max_y = 1.0;
            }

            let mut trigger = false;
            let plot_px_w = ui.available_width() as usize;

            // --- HEADER ---
            ui.horizontal(|ui| {
                ui.heading(&self.title);

                // Measure status label width once and cache it so the slider
                // position doesn't shift when the status string changes.
                let max_status_w = *self.status_label_width.get_or_insert_with(|| {
                    let font_id = TextStyle::Body.resolve(ui.style());
                    ui.fonts(|f| {
                        ["• LIVE", "EXPLORE", "⚠️ Ended"]
                            .iter()
                            .map(|s| {
                                f.layout_no_wrap(s.to_string(), font_id.clone(), Color32::WHITE)
                                    .rect
                                    .width()
                            })
                            .fold(0.0f32, f32::max)
                    })
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
                let min_zx = (self.default_width / f64::max(last_x - first_x, self.default_width))
                    .max(0.0001);
                if ui
                    .add_sized(
                        [250.0, 20.0],
                        egui::Slider::new(&mut self.x_zoom, min_zx..=2000.0)
                            .show_value(false)
                            .logarithmic(true),
                    )
                    .changed()
                {
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
                ui.separator();
                let btn_label = if self.identifying {
                    "Cancel Point ID"
                } else {
                    "Identify Point"
                };
                if ui.toggle_value(&mut self.identifying, btn_label).clicked() && self.identifying {
                    self.identified = None; // clear old result when entering identify mode
                }
                if let Some((si, ix, iy)) = self.identified {
                    let xf = format_x_val(ix, self.is_ts_mode);
                    ui.label(
                        egui::RichText::new(format!(
                            "{} ➔ Y: {:.4} | X: {}",
                            self.labels[si], iy, xf
                        ))
                        .color(self.colors[si])
                        .strong(),
                    );
                } else if self.identifying {
                    ui.label(
                        egui::RichText::new("Click a point on the plot...")
                            .weak()
                            .italics(),
                    );
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
                    let min_zy = (self.y_nat_h / f64::max(g_max_y - g_min_y, 0.1)).min(1.0);
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
                        let local_offset = DateTime::from_timestamp(input.bounds.0 as i64, 0)
                            .map(|dt| {
                                dt.with_timezone(&Local).offset().fix().local_minus_utc() as f64
                            })
                            .unwrap_or(0.0);
                        let start_aligned = ((input.bounds.0 + local_offset) / minor_step).ceil()
                            * minor_step
                            - local_offset;
                        let mut marks = Vec::new();
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
                        let local_offset = DateTime::from_timestamp(mark.value as i64, 0)
                            .map(|dt| {
                                dt.with_timezone(&Local).offset().fix().local_minus_utc() as f64
                            })
                            .unwrap_or(0.0);
                        if ((mark.value + local_offset) % mark.step_size).abs() < 1e-5 {
                            DateTime::from_timestamp(mark.value as i64, 0)
                                .map(|dt| {
                                    let ldt = dt.with_timezone(&Local);
                                    let span = *range.end() - *range.start();
                                    if span > 86400.0 {
                                        ldt.format("%m/%d %H:%M").to_string()
                                    } else if mark.step_size >= 60.0 {
                                        ldt.format("%H:%M").to_string()
                                    } else {
                                        ldt.format("%H:%M:%S").to_string()
                                    }
                                })
                                .unwrap_or_default()
                        } else {
                            String::new()
                        }
                    });
                } else {
                    plot = plot.x_axis_formatter(move |m, _| format!("{:.0}", m.value));
                }

                for &y in &self.include_y_values {
                    plot = plot.include_y(y);
                }

                let data_arc = self.smoothed_data.clone();
                let labels = self.labels.clone();
                let colors = self.colors.clone();
                let include_y = self.include_y_values.clone();
                let visible = self.visible.clone();
                let def_w = self.default_width;

                let mut click_mp: Option<egui_plot::PlotPoint> = None;
                let mut current_bounds: Option<PlotBounds> = None;

                let plot_res = plot.show(ui, |plot_ui| {
                    current_bounds = Some(plot_ui.plot_bounds());
                    if self.identifying && plot_ui.ctx().input(|i| i.pointer.primary_clicked()) {
                        click_mp = plot_ui.pointer_coordinate();
                    }
                    if self.identifying && plot_ui.response().hovered() {
                        plot_ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                    }
                    let is_interacting = plot_ui.pointer_coordinate_drag_delta().length() > 0.0
                        || plot_ui.ctx().input(|i| i.raw_scroll_delta.y).abs() > 0.0;
                    if is_interacting {
                        self.auto_follow = false;
                        self.interacting_frames = 6; // stay coarse ~100ms after last event
                    } else if self.interacting_frames > 0 {
                        self.interacting_frames -= 1;
                        ctx.request_repaint(); // drive countdown to zero for final full-res frame
                    }
                    let coarse = self.interacting_frames > 0;

                    let b = plot_ui.plot_bounds();

                    // Double-click: zoom to show all data.
                    if plot_ui.ctx().input(|i| {
                        i.pointer
                            .button_double_clicked(egui::PointerButton::Primary)
                    }) && plot_ui.response().hovered()
                    {
                        self.auto_follow = false;
                        trigger = true;
                        self.scroll_offset = first_x;
                        self.x_zoom = def_w / f64::max(last_x - first_x, 0.001);
                        self.y_min = g_min_y;
                        self.y_max = g_max_y;
                        self.y_nat_h = f64::max(g_max_y - g_min_y, 0.001);
                        self.y_zoom = 1.0;
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [first_x, g_min_y],
                            [last_x, g_max_y],
                        ));
                    }

                    if self.auto_follow {
                        let width = def_w / self.x_zoom;
                        let x_start = last_x - width;
                        let mut min_y_vis = f64::INFINITY;
                        let mut max_y_vis = f64::NEG_INFINITY;
                        let d = data_arc.lock().unwrap();
                        for (i, buf) in d.iter().enumerate() {
                            if !visible[i] || buf.is_empty() {
                                continue;
                            }
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
                        self.y_nat_h = f64::max(base.1 - base.0, 0.001);
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

                    // --- CACHED M4 BINNING ---
                    // Use coarser buckets during interaction for responsiveness;
                    // revert to full pixel-width resolution when settled.
                    let full_bins = plot_px_w.max(16);
                    let bins = if coarse {
                        (full_bins / 4).max(16)
                    } else {
                        full_bins
                    };

                    let data_changed = self.new_data.swap(false, Ordering::Relaxed);
                    let cur_view = [b.min()[0], b.min()[1], b.max()[0], b.max()[1]];

                    // Invalidate cache when: new data, viewport moved, bucket count changed,
                    // or cache is empty (first frame / after visibility/tau change).
                    let need_recompute = data_changed
                        || self.cached_lines.is_empty()
                        || self
                            .last_view_rect
                            .map_or(true, |r| r != cur_view || coarse != (bins < full_bins));

                    if need_recompute {
                        let d = data_arc.lock().unwrap();
                        self.cached_lines.clear();
                        for (i, buf) in d.iter().enumerate() {
                            if !visible[i] || buf.is_empty() {
                                self.cached_lines.push(Vec::new());
                                continue;
                            }
                            let s_idx = buf.search_x(b.min()[0]).saturating_sub(1);
                            // Stable tail: if the right viewport edge is past the last sample,
                            // include all remaining points rather than truncating.
                            let e_idx = if b.max()[0] >= buf.last().unwrap()[0] {
                                buf.len()
                            } else {
                                buf.search_x(b.max()[0]).min(buf.len())
                            };
                            self.cached_lines.push(m4(
                                buf,
                                s_idx,
                                e_idx,
                                bins,
                                b.min()[0],
                                b.max()[0],
                            ));
                        }
                        self.last_view_rect = Some(cur_view);
                        self.view_settling = true;
                        ctx.request_repaint();
                    } else if self.view_settling {
                        self.view_settling = false;
                        ctx.request_repaint();
                    }

                    // --- HIGHLIGHT-AWARE LINE RENDERING ---
                    let any_highlighted = self.highlighted.iter().any(|&h| h);
                    for (i, pts) in self.cached_lines.iter().enumerate() {
                        if !visible[i] || pts.is_empty() {
                            continue;
                        }
                        let base_color = colors[i];
                        let (line_color, line_width) = if any_highlighted {
                            if self.highlighted[i] {
                                (base_color, 2.5f32)
                            } else {
                                (base_color.linear_multiply(0.25), 1.0f32)
                            }
                        } else {
                            (base_color, 1.5f32)
                        };
                        plot_ui.line(
                            Line::new(PlotPoints::new(pts.clone()))
                                .name(&labels[i])
                                .color(line_color)
                                .width(line_width),
                        );
                    }

                    // --- IDENTIFY: crosshair on identified point ---
                    if let Some((si, ix, iy)) = self.identified {
                        let pt_color = colors[si];
                        plot_ui.vline(VLine::new(ix).color(pt_color).width(1.0));
                        plot_ui.hline(HLine::new(iy).color(pt_color).width(1.0));
                        // Bright dot at the exact point
                        plot_ui.points(
                            Points::new(PlotPoints::new(vec![[ix, iy]]))
                                .color(Color32::WHITE)
                                .radius(5.0),
                        );
                    }
                });

                // Identify search runs outside the plot closure so response().clicked()
                // is reliable and doesn't fire on drag-release.
                if self.identifying && plot_res.response.clicked() {
                    if let (Some(mp), Some(b)) = (click_mp, current_bounds) {
                        let d = data_arc.lock().unwrap();
                        let y_scale = b.width() / b.height().max(0.1);
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
                                for j in idx.saturating_sub(2)..(idx + 3).min(buf.len()) {
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
                        // Only snap if click is within 5% of view width of a point
                        if best_dsq < (b.width() * 0.05).powi(2) {
                            if let Some((si, px, py)) = best {
                                self.identified = Some((si, px, py));
                                self.highlighted[si] = true;
                            }
                        }
                        self.identifying = false;
                        ctx.request_repaint();
                    }
                }

                // Floating custom legend overlay on the plot
                if let Some(pos) = self.legend_corner {
                    let rect = plot_res.response.rect;
                    let pad = 10.0;
                    let (align, anchor_pos) = match pos {
                        Corner::LeftTop => (
                            egui::Align2::LEFT_TOP,
                            rect.left_top() + egui::vec2(pad, pad),
                        ),
                        Corner::RightTop => (
                            egui::Align2::RIGHT_TOP,
                            rect.right_top() + egui::vec2(-pad, pad),
                        ),
                        Corner::LeftBottom => (
                            egui::Align2::LEFT_BOTTOM,
                            rect.left_bottom() + egui::vec2(pad, -pad),
                        ),
                        Corner::RightBottom => (
                            egui::Align2::RIGHT_BOTTOM,
                            rect.right_bottom() + egui::vec2(-pad, -pad),
                        ),
                    };
                    egui::Window::new("CustomLegend")
                        .id(egui::Id::new("custom_legend_window"))
                        .fixed_pos(anchor_pos)
                        .pivot(align)
                        .title_bar(false)
                        .resizable(false)
                        .collapsible(false)
                        .frame(
                            egui::Frame::window(&ctx.style())
                                .fill(Color32::from_black_alpha(200))
                                .inner_margin(8.0),
                        )
                        .show(ctx, |ui| {
                            let any_highlighted = self.highlighted.iter().any(|&h| h);
                            let mut item_clicked = None;
                            for i in 0..self.labels.len() {
                                if !self.visible[i] {
                                    continue;
                                }
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(
                                        egui::vec2(12.0, 12.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(r, 2.0, self.colors[i]);
                                    let is_hl = self.highlighted[i];
                                    let text_color = if is_hl {
                                        Color32::WHITE
                                    } else if any_highlighted {
                                        Color32::GRAY
                                    } else {
                                        ctx.style().visuals.text_color()
                                    };
                                    if ui
                                        .selectable_label(
                                            is_hl,
                                            egui::RichText::new(&self.labels[i]).color(text_color),
                                        )
                                        .clicked()
                                    {
                                        item_clicked = Some(i);
                                    }
                                });
                            }
                            // Click on empty legend background clears highlight
                            if ui.input(|i| i.pointer.primary_clicked()) && item_clicked.is_none() {
                                if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
                                    if ui.min_rect().contains(pos) {
                                        self.highlighted.iter_mut().for_each(|h| *h = false);
                                        self.identified = None;
                                    }
                                }
                            }
                            if let Some(i) = item_clicked {
                                self.highlighted[i] = !self.highlighted[i];
                                if !self.highlighted.iter().any(|&h| h) {
                                    self.identified = None;
                                }
                            }
                        });
                }

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
    let mut raw_labels = args
        .labels
        .clone()
        .unwrap_or_else(|| vec!["Series 1".to_string()]);
    let label_count = raw_labels.len();
    if args.sort_labels {
        raw_labels.sort();
    }
    let display_labels = raw_labels;

    let input_to_display_map: Vec<usize> = {
        let mut map = vec![0usize; label_count];
        let original_labels = args
            .labels
            .clone()
            .unwrap_or_else(|| vec!["Series 1".to_string()]);
        for (orig_idx, orig_label) in original_labels.iter().enumerate() {
            if let Some(disp_idx) = display_labels.iter().position(|l| l == orig_label) {
                map[orig_idx] = disp_idx;
            }
        }
        map
    };

    let raw_buffer: Arc<Mutex<Vec<SeriesBuffer>>> = Arc::new(Mutex::new(
        (0..label_count)
            .map(|_| SeriesBuffer::new(args.max_points))
            .collect(),
    ));
    let smooth_buffer: Arc<Mutex<Vec<SeriesBuffer>>> = Arc::new(Mutex::new(
        (0..label_count)
            .map(|_| SeriesBuffer::new(args.max_points))
            .collect(),
    ));
    let tau_shared = Arc::new(Mutex::new(0.000001f64));
    let stream_ended = Arc::new(AtomicBool::new(false));
    let new_data = Arc::new(AtomicBool::new(false));

    let (raw_thread, smooth_thread, tau_thread, stream_ended_thread, new_data_thread) = (
        Arc::clone(&raw_buffer),
        Arc::clone(&smooth_buffer),
        Arc::clone(&tau_shared),
        Arc::clone(&stream_ended),
        Arc::clone(&new_data),
    );

    eframe::run_native(
        "Live Plotter",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 850.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            let app_args = args.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                let mut seq = 0u64;
                for (li, line) in stdin.lock().lines().enumerate() {
                    if let Ok(line_str) = line {
                        let trimmed = line_str.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let tokens: Vec<&str> = trimmed
                            .split(|c| c == ',' || c == ' ')
                            .filter(|s| !s.is_empty())
                            .collect();
                        let expected = if is_ts { label_count + 1 } else { label_count };
                        if tokens.len() != expected {
                            eprintln!(
                                "FATAL ERROR (line {}): expected {} cols, found {}.",
                                li + 1,
                                expected,
                                tokens.len()
                            );
                            std::process::exit(1);
                        }
                        let (x, data_tokens) = if is_ts {
                            (tokens[0].parse::<f64>().expect("Time fail"), &tokens[1..])
                        } else {
                            (seq as f64 * period, &tokens[..])
                        };
                        {
                            let mut rb = raw_thread.lock().unwrap();
                            let mut sb = smooth_thread.lock().unwrap();
                            let t = *tau_thread.lock().unwrap();
                            for i in 0..label_count {
                                let disp_i = input_to_display_map[i];
                                let tok = data_tokens[i].to_ascii_lowercase();
                                if tok == "none" || tok == "null" {
                                    continue;
                                }
                                if let Ok(v) = data_tokens[i].parse::<f64>() {
                                    rb[disp_i].push([x, v]);
                                    if is_ts {
                                        let mut j = rb[disp_i].len() - 1;
                                        while j > 0
                                            && rb[disp_i].data[j][0] < rb[disp_i].data[j - 1][0]
                                        {
                                            rb[disp_i].data.swap(j, j - 1);
                                            j -= 1;
                                        }
                                    }
                                    let y = if let Some(last) = sb[disp_i].last() {
                                        let dt = (x - last[0]).max(0.0);
                                        if t <= 1e-6 {
                                            v
                                        } else {
                                            let alpha = 1.0 - (-(dt / t)).exp();
                                            alpha * v + (1.0 - alpha) * last[1]
                                        }
                                    } else {
                                        v
                                    };
                                    sb[disp_i].push([x, y]);
                                }
                            }
                            seq += 1;
                        }
                        new_data_thread.store(true, Ordering::Relaxed);
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
                new_data,
                display_labels,
            )))
        }),
    )
    .unwrap();
}
