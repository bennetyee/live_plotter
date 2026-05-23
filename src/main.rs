use chrono::{Local, TimeZone};
use clap::Parser;
use eframe::egui;
use egui::{Color32, TextStyle};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints};
use std::io::{self, BufRead};
use std::process;
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
}

type SeriesBuffer = Vec<[f64; 2]>;

struct LivePlotApp {
    data: Arc<Mutex<Vec<SeriesBuffer>>>,
    stream_ended: Arc<AtomicBool>,
    include_y_values: Vec<f64>,
    labels: Vec<String>,
    colors: Vec<Color32>,
    visible: Vec<bool>,
    title: String,
    legend_corner: Option<Corner>,
    is_ts_mode: bool,

    // UI State
    side_panel_collapsed: bool,
    default_width: f64,
    x_zoom: f64,
    x_center: f64,
    y_zoom: f64,
    y_center: f64,
    y_nat_h: f64,
    y_min: f64,
    y_max: f64,
    scroll_offset: f64,
    auto_follow: bool,
}

impl LivePlotApp {
    fn new(
        args: Args,
        data: Arc<Mutex<Vec<SeriesBuffer>>>,
        stream_ended: Arc<AtomicBool>,
        labels: Vec<String>,
    ) -> Self {
        let visible = vec![true; labels.len()];
        let legend_corner = match args.legend_pos.to_lowercase().as_str() {
            "none" => None,
            "lefttop" => Some(Corner::LeftTop),
            "righttop" => Some(Corner::RightTop),
            "leftbottom" => Some(Corner::LeftBottom),
            "rightbottom" => Some(Corner::RightBottom),
            _ => {
                eprintln!("FATAL ERROR: Invalid --legend-pos '{}'.", args.legend_pos);
                eprintln!("Valid values: LeftTop, RightTop, LeftBottom, RightBottom, None");
                process::exit(1);
            }
        };

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
        if let Some(h_list) = args.colors {
            for hex in h_list {
                if let Ok(c) = Color32::from_hex(&hex) {
                    colors.push(c);
                }
            }
        }
        for i in colors.len()..labels.len() {
            colors.push(palette[i % palette.len()]);
        }

        Self {
            data,
            stream_ended,
            include_y_values: args.include_y,
            labels,
            colors,
            visible,
            title: args.title,
            legend_corner,
            is_ts_mode: args.timestamp,
            side_panel_collapsed: false,
            default_width: args.viewport_width,
            x_zoom: 1.0,
            x_center: 0.0,
            y_zoom: 1.0,
            y_center: 0.0,
            y_nat_h: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            scroll_offset: 0.0,
            auto_follow: true,
        }
    }
}

fn format_x_val(x: f64, is_ts: bool) -> String {
    if is_ts {
        if let Some(dt) = Local
            .timestamp_opt(x as i64, ((x % 1.0) * 1e9) as u32)
            .single()
        {
            return dt.format("%H:%M:%S").to_string();
        }
    }
    format!("{:.0}", x)
}

impl eframe::App for LivePlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- 1. DYNAMIC PANEL WIDTH CALCULATION ---
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
            (max_label_w + 85.0).max(180.0).min(600.0)
        };

        egui::SidePanel::right("vis_panel")
            .resizable(false)
            .exact_width(panel_width)
            .show(ctx, |ui| {
                if self.side_panel_collapsed {
                    ui.vertical_centered(|ui| {
                        ui.add_space(10.0);
                        if ui.button("⏴").on_hover_text("Expand").clicked() {
                            self.side_panel_collapsed = false;
                        }
                    });
                } else {
                    ui.horizontal(|ui| {
                        if ui.button("⏵").on_hover_text("Collapse").clicked() {
                            self.side_panel_collapsed = true;
                        }
                        ui.heading("Visibility");
                    });
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

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut first_x = 0.0;
            let mut last_x = 0.0;
            {
                let d = self.data.lock().unwrap();
                if let Some(b) = d.get(0) {
                    if let (Some(f), Some(l)) = (b.first(), b.last()) {
                        first_x = f[0];
                        last_x = l[0];
                    }
                }
            }

            let mut trigger = false;
            let layer_id = ui.layer_id();

            // --- 2. HEADER ---
            ui.horizontal(|ui| {
                ui.heading(&self.title);
                if self.stream_ended.load(Ordering::Relaxed) {
                    ui.colored_label(Color32::GOLD, "⚠️ Ended");
                } else if self.auto_follow {
                    ui.colored_label(Color32::from_rgb(100, 200, 255), "• LIVE");
                }
                ui.separator();
                ui.label("X-Zoom:");

                let cur_x_center = self.x_center;
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
                    self.x_center = cur_x_center;
                }
                if ui.button("Reset Viewport").clicked() {
                    self.x_zoom = 1.0;
                    self.y_zoom = 1.0;
                    self.auto_follow = true;
                    trigger = true;
                }
            });

            // Local shadowing of state for the plot closure to handle independent zoom/centers
            let data_arc = self.data.clone();
            let is_ts = self.is_ts_mode;
            let include_y = self.include_y_values.clone();
            let labels = self.labels.clone();
            let colors = self.colors.clone();
            let visible = self.visible.clone();
            let def_w = self.default_width;

            let mut auto_follow = self.auto_follow;
            let mut x_zoom = self.x_zoom;
            let mut x_center = self.x_center;
            let mut y_zoom = self.y_zoom;
            let mut y_center = self.y_center;
            let mut y_min = self.y_min;
            let mut y_max = self.y_max;
            let mut y_nat_h = self.y_nat_h;
            let mut scroll_offset = self.scroll_offset;

            // --- 3. BODY (STRETCHED) ---
            let body_layout =
                egui::Layout::left_to_right(egui::Align::Min).with_cross_justify(true);
            ui.with_layout(body_layout, |ui| {
                let cur_y_h = (y_max - y_min).max(0.0001);
                let cur_y_center = y_min + (cur_y_h / 2.0);

                let vy_resp = ui.add_sized(
                    [20.0, ui.available_height()],
                    egui::Slider::new(&mut y_zoom, 0.5..=10.0)
                        .vertical()
                        .show_value(false)
                        .logarithmic(true),
                );
                if vy_resp.changed() {
                    auto_follow = false;
                    trigger = true;
                }

                let mut plot = Plot::new("lp")
                    .height(ui.available_height())
                    .width(ui.available_width())
                    .auto_bounds([false, false].into())
                    .allow_zoom(true)
                    .allow_drag(true)
                    .show_x(false)
                    .show_y(false)
                    .label_formatter(|_, _| String::new())
                    .x_axis_formatter({
                        let is_ts = self.is_ts_mode;
                        move |m, _| format_x_val(m.value, is_ts)
                    });

                if let Some(pos) = self.legend_corner {
                    plot = plot.legend(Legend::default().position(pos));
                }
                for &y in &include_y {
                    plot = plot.include_y(y);
                }

                let plot_res = plot.show(ui, |plot_ui| {
                    if plot_ui.pointer_coordinate_drag_delta().length() > 0.0
                        || plot_ui.ctx().input(|i| i.raw_scroll_delta.y).abs() > 0.0
                    {
                        auto_follow = false;
                    }

                    if auto_follow {
                        let width = def_w / x_zoom;
                        let x_start = last_x - width;
                        x_center = last_x - (width / 2.0);

                        let mut min_y_vis = f64::INFINITY;
                        let mut max_y_vis = f64::NEG_INFINITY;
                        let d = data_arc.lock().unwrap();
                        for (i, b) in d.iter().enumerate() {
                            if !visible[i] {
                                continue;
                            }
                            for p in b.iter().filter(|p| p[0] >= x_start) {
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
                        y_center = (base.0 + base.1) / 2.0;
                        y_nat_h = (base.1 - base.0).max(0.001);
                        y_zoom = 1.0;
                        y_min = base.0;
                        y_max = base.1;
                        scroll_offset = x_start;
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [x_start, base.0],
                            [last_x, base.1],
                        ));
                    } else if trigger {
                        let hw = (def_w / x_zoom) / 2.0;
                        let hh = (y_nat_h / y_zoom) / 2.0;
                        let final_y_min = cur_y_center - hh;
                        let final_y_max = cur_y_center + hh;
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [x_center - hw, final_y_min],
                            [x_center + hw, final_y_max],
                        ));
                        y_min = final_y_min;
                        y_max = final_y_max;
                    } else {
                        let b = plot_ui.plot_bounds();
                        x_center = b.center().x;
                        y_center = b.center().y;
                        y_min = b.min()[1];
                        y_max = b.max()[1];
                        scroll_offset = b.min()[0];
                        if b.width() > 0.0 {
                            x_zoom = def_w / b.width();
                        }
                        if b.height() > 0.0 {
                            y_zoom = (y_nat_h / b.height()).max(0.001);
                        }
                    }

                    let d = data_arc.lock().unwrap();
                    for (i, b) in d.iter().enumerate() {
                        if !visible[i] {
                            continue;
                        }
                        let slen = b.len();
                        if slen == 0 {
                            continue;
                        }
                        plot_ui.line(
                            Line::new(PlotPoints::from_parametric_callback(
                                move |t| {
                                    let idx = t.round() as usize;
                                    if idx < slen {
                                        (b[idx][0], b[idx][1])
                                    } else {
                                        (0.0, 0.0)
                                    }
                                },
                                0.0..=(slen as f64 - 1.0),
                                slen,
                            ))
                            .name(&labels[i])
                            .color(colors[i]),
                        );
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
                });

                self.auto_follow = auto_follow;
                self.x_zoom = x_zoom;
                self.x_center = x_center;
                self.y_zoom = y_zoom;
                self.y_center = y_center;
                self.y_min = y_min;
                self.y_max = y_max;
                self.y_nat_h = y_nat_h;
                self.scroll_offset = scroll_offset;

                if plot_res.response.dragged() || trigger {
                    ui.ctx().request_repaint();
                }
            });
        });
    }
}

fn main() {
    let args = Args::parse();
    let is_ts = args.timestamp;
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
    let data_buffer = Arc::new(Mutex::new(vec![Vec::new(); label_count]));
    let data_buffer_thread = Arc::clone(&data_buffer);
    let stream_ended = Arc::new(AtomicBool::new(false));
    let stream_ended_thread = Arc::clone(&stream_ended);
    let max_pts = args.max_points;
    let app_args = args.clone();

    eframe::run_native(
        "Live Plotter",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1250.0, 750.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                let mut seq = 0;
                for (_li, line) in stdin.lock().lines().enumerate() {
                    if let Ok(line_str) = line {
                        let trimmed = line_str.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let vals: Vec<f64> = trimmed
                            .split(|c| c == ',' || c == ' ')
                            .filter(|s| !s.is_empty())
                            .filter_map(|s| s.parse().ok())
                            .collect();
                        if vals.len() != expected {
                            eprintln!("FATAL ERROR: schema mismatch.");
                            process::exit(1);
                        }
                        let (x, series) = if is_ts {
                            (vals[0], vals[1..].to_vec())
                        } else {
                            (seq as f64, vals)
                        };
                        {
                            let mut b = data_buffer_thread.lock().unwrap();
                            for (original_i, v) in series.into_iter().enumerate() {
                                let display_i = input_to_display_map[original_i];
                                if display_i < b.len() {
                                    if b[display_i].len() >= max_pts {
                                        b[display_i].remove(0);
                                    }
                                    b[display_i].push([x, v]);
                                    if is_ts {
                                        let mut j = b[display_i].len() - 1;
                                        while j > 0 && b[display_i][j][0] < b[display_i][j - 1][0] {
                                            b[display_i].swap(j, j - 1);
                                            j -= 1;
                                        }
                                    }
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
                data_buffer,
                stream_ended,
                display_labels,
            )))
        }),
    )
    .unwrap();
}
