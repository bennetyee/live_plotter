use chrono::{DateTime, Local, Offset};
use clap::Parser;
use eframe::egui;
use egui::{Color32, TextStyle};
use egui_plot::{Corner, GridInput, GridMark, Legend, Line, Plot, PlotBounds, PlotPoints};
use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "High-Performance Event-Driven Plotter")]
struct Args {
    #[arg(short, long, default_value_t = 1000000)]
    max_points: usize,
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
    #[arg(long, default_value = "LeftTop")]
    legend_pos: String,
    #[arg(long, default_value_t = 60.0)]
    max_tau: f64,
}

type SeriesBuffer = VecDeque<[f64; 2]>;

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
            smooth_series.clear();
            if let Some(first) = raw_series.front() {
                smooth_series.push_back(*first);
                let (mut prev_y, mut prev_t) = (first[1], first[0]);
                for point in raw_series.iter().skip(1) {
                    let (cur_t, cur_x) = (point[0], point[1]);
                    let alpha = 1.0 - (-(cur_t - prev_t).max(0.0) / self.tau).exp();
                    let y = if self.tau <= 1e-6 {
                        cur_x
                    } else {
                        alpha * cur_x + (1.0 - alpha) * prev_y
                    };
                    smooth_series.push_back([cur_t, y]);
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
        // --- SIDE PANEL ---
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

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut first_x = 0.0;
            let mut last_x = 0.0;
            let mut g_min_y = f64::INFINITY;
            let mut g_max_y = f64::NEG_INFINITY;
            {
                let d = self.raw_data.lock().unwrap();
                let mut found_x = false;
                for buf in d.iter() {
                    if let (Some(f), Some(l)) = (buf.front(), buf.back()) {
                        if !found_x || f[0] < first_x {
                            first_x = f[0];
                        }
                        if !found_x || l[0] > last_x {
                            last_x = l[0];
                        }
                        found_x = true;
                    }
                    for p in buf.iter() {
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
            let layer_id = ui.layer_id();
            let plot_px_w = ui.available_width();

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
                let min_zx =
                    (self.default_width / (last_x - first_x).max(self.default_width)).max(0.0001);
                if ui
                    .add_sized(
                        [(ui.available_width() - 115.0).max(50.0), 20.0],
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
                    let min_zy = (self.y_nat_h / (g_max_y - g_min_y).max(0.1)).min(1.0);
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
                        let local_offset =
                            if let Some(dt) = DateTime::from_timestamp(input.bounds.0 as i64, 0) {
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
                    let drag_delta = plot_ui.pointer_coordinate_drag_delta();
                    let scroll_delta = plot_ui.ctx().input(|i| i.raw_scroll_delta.y);
                    if drag_delta.length() > 0.0 || scroll_delta.abs() > 0.0 {
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
                            if !visible[i] || buf.is_empty() {
                                continue;
                            }
                            // Binary search for visible range
                            let mut l = 0;
                            let mut h = buf.len();
                            while l < h {
                                let m = l + (h - l) / 2;
                                if buf[m][0] < x_start {
                                    l = m + 1;
                                } else {
                                    h = m;
                                }
                            }
                            for p in buf.iter().skip(l.saturating_sub(1)) {
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

                    // --- STABLE M4 BINNING ---
                    let bins = plot_px_w as usize;
                    let d = data_arc.lock().unwrap();
                    for (i, buf) in d.iter().enumerate() {
                        if !visible[i] || buf.is_empty() {
                            continue;
                        }

                        let (mut l, mut h) = (0, buf.len());
                        while l < h {
                            let m = l + (h - l) / 2;
                            if buf[m][0] < b.min()[0] {
                                l = m + 1;
                            } else {
                                h = m;
                            }
                        }
                        let s_idx = l.saturating_sub(1);
                        let (mut l, mut h) = (0, buf.len());
                        while l < h {
                            let m = l + (h - l) / 2;
                            if buf[m][0] < b.max()[0] {
                                l = m + 1;
                            } else {
                                h = m;
                            }
                        }
                        // Fix flicker: Force e_idx to include the very last point if the viewport includes the end
                        let e_idx = if b.max()[0] >= buf.back().unwrap()[0] {
                            buf.len()
                        } else {
                            l.min(buf.len())
                        };

                        let count = e_idx - s_idx;
                        if count == 0 {
                            continue;
                        }

                        if count > bins * 4 {
                            let step = count / bins;
                            plot_ui.line(
                                Line::new(PlotPoints::from_parametric_callback(
                                    move |t| {
                                        let bin_idx = (t.round() as usize) / 4;
                                        let sub_idx = (t.round() as usize) % 4;
                                        let bs = s_idx + (bin_idx * step);
                                        let be = (bs + step).min(e_idx);
                                        if bs >= buf.len() {
                                            return (buf[buf.len() - 1][0], buf[buf.len() - 1][1]);
                                        }
                                        match sub_idx {
                                            0 => (buf[bs][0], buf[bs][1]),
                                            3 => (buf[be - 1][0], buf[be - 1][1]),
                                            _ => {
                                                let mut l_min = f64::INFINITY;
                                                let mut l_max = f64::NEG_INFINITY;
                                                let (mut mi, mut mai) = (bs, bs);
                                                for j in bs..be {
                                                    let y = buf[j][1];
                                                    if y < l_min {
                                                        l_min = y;
                                                        mi = j;
                                                    }
                                                    if y > l_max {
                                                        l_max = y;
                                                        mai = j;
                                                    }
                                                }
                                                let (early, late) =
                                                    if mi < mai { (mi, mai) } else { (mai, mi) };
                                                let p = if sub_idx == 1 {
                                                    buf[early]
                                                } else {
                                                    buf[late]
                                                };
                                                (p[0], p[1])
                                            }
                                        }
                                    },
                                    0.0..=((bins * 4) as f64),
                                    (bins * 4) + 1,
                                ))
                                .name(&labels[i])
                                .color(colors[i]),
                            );
                        } else {
                            plot_ui.line(
                                Line::new(PlotPoints::from_parametric_callback(
                                    move |t| {
                                        let idx = (s_idx + t.round() as usize).min(e_idx - 1);
                                        (buf[idx][0], buf[idx][1])
                                    },
                                    0.0..=(count as f64 - 1.0),
                                    count,
                                ))
                                .name(&labels[i])
                                .color(colors[i]),
                            );
                        }
                    }

                    if let Some(mp) = plot_ui.pointer_coordinate() {
                        let mut best = None;
                        let mut bd = f64::INFINITY;
                        for si in 0..labels.len() {
                            if !visible[si] {
                                continue;
                            }
                            if let Some(buf) = d.get(si) {
                                if buf.is_empty() {
                                    continue;
                                }
                                let (mut l, mut h) = (0, buf.len());
                                while l < h {
                                    let m = l + (h - l) / 2;
                                    if buf[m][0] < mp.x {
                                        l = m + 1;
                                    } else {
                                        h = m;
                                    }
                                }
                                for j in l.saturating_sub(1)..(l + 2).min(buf.len()) {
                                    let p = buf[j];
                                    let dx = p[0] - mp.x;
                                    let dy = (p[1] - mp.y) * (b.width() / b.height().max(0.1));
                                    let dsq = dx * dx + dy * dy;
                                    if dsq < bd {
                                        bd = dsq;
                                        best = Some((si, p[0], p[1]));
                                    }
                                }
                            }
                        }
                        if let Some((si, x, y)) = best {
                            if bd < (b.width() * 0.015).powi(2) {
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

                    let cur_v = [b.min()[0], b.min()[1], b.max()[0], b.max()[1]];
                    if self.last_view_rect != Some(cur_v) {
                        self.last_view_rect = Some(cur_v);
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
        let mut map = vec![0; label_count];
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

    let expected = if is_ts { label_count + 1 } else { label_count };
    let raw_buffer: Arc<Mutex<Vec<SeriesBuffer>>> = Arc::new(Mutex::new(
        (0..label_count)
            .map(|_| VecDeque::with_capacity(args.max_points))
            .collect(),
    ));
    let smooth_buffer: Arc<Mutex<Vec<SeriesBuffer>>> = Arc::new(Mutex::new(
        (0..label_count)
            .map(|_| VecDeque::with_capacity(args.max_points))
            .collect(),
    ));

    let raw_thread = Arc::clone(&raw_buffer);
    let smooth_thread = Arc::clone(&smooth_buffer);
    let tau_shared = Arc::new(Mutex::new(0.000001));
    let tau_thread = Arc::clone(&tau_shared);
    let stream_ended = Arc::new(AtomicBool::new(false));
    let stream_ended_thread = Arc::clone(&stream_ended);
    let max_pts = args.max_points;

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
                            for i in 0..label_count {
                                if let Ok(v) = data_tokens[i].parse::<f64>() {
                                    if rb[i].len() >= max_pts {
                                        rb[i].pop_front();
                                    }
                                    rb[i].push_back([x, v]);
                                    if is_ts {
                                        let mut j = rb[i].len() - 1;
                                        while j > 0 && rb[i][j][0] < rb[i][j - 1][0] {
                                            rb[i].swap(j, j - 1);
                                            j -= 1;
                                        }
                                    }
                                    if sb[i].len() >= max_pts {
                                        sb[i].pop_front();
                                    }
                                    let y = if let Some(last) = sb[i].back() {
                                        let dt = (x - last[0]).max(0.0);
                                        let alpha = 1.0 - (-(dt / t)).exp();
                                        if t <= 1e-6 {
                                            v
                                        } else {
                                            alpha * v + (1.0 - alpha) * last[1]
                                        }
                                    } else {
                                        v
                                    };
                                    sb[i].push_back([x, y]);
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
                args.clone(),
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
