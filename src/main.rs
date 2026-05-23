use chrono::{Local, TimeZone};
use clap::Parser;
use eframe::egui;
use egui::Color32;
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints};
use std::io::{self, BufRead};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Zero-Copy High-Performance Live Plotter")]
struct Args {
    #[arg(short, long, default_value_t = 10000)]
    max_points: usize,
    #[arg(short, long, default_value_t = 500.0)]
    viewport_width: f64,
    #[arg(short, long, default_value_t = false)]
    timestamp: bool,
    #[arg(long, num_args = 1..)]
    include_y: Vec<f64>,
    #[arg(short, long, num_args = 1..)]
    labels: Option<Vec<String>>,
    #[arg(short, long, num_args = 1..)]
    colors: Option<Vec<String>>,
    #[arg(long, default_value = "Live Time-Series Feed")]
    title: String,
    #[arg(long, default_value = "LeftTop")]
    legend_pos: String,
}

// Storage for Zero-Copy rendering (contiguous floats)
type SeriesBuffer = Vec<[f64; 2]>;

struct LivePlotApp {
    data: Arc<Mutex<Vec<SeriesBuffer>>>,
    stream_ended: Arc<AtomicBool>,
    include_y_values: Vec<f64>,
    labels: Vec<String>,
    colors: Vec<Color32>,
    title: String,
    legend_corner: Corner,
    is_ts_mode: bool,

    default_width: f64,
    zoom_factor: f64,
    y_zoom_factor: f64,
    scroll_offset: f64,
    auto_follow: bool,
}

impl LivePlotApp {
    fn new(args: Args, data: Arc<Mutex<Vec<SeriesBuffer>>>, stream_ended: Arc<AtomicBool>) -> Self {
        let labels = args.labels.unwrap_or_else(|| vec!["Series 1".to_string()]);

        let legend_corner = match args.legend_pos.to_lowercase().as_str() {
            "lefttop" => Corner::LeftTop,
            "righttop" => Corner::RightTop,
            "leftbottom" => Corner::LeftBottom,
            "rightbottom" => Corner::RightBottom,
            _ => {
                eprintln!("FATAL ERROR: Invalid --legend-pos '{}'.", args.legend_pos);
                eprintln!("Accepted values: LeftTop, RightTop, LeftBottom, RightBottom");
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
        if let Some(hex_list) = args.colors {
            for hex in hex_list {
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
            title: args.title,
            legend_corner,
            is_ts_mode: args.timestamp,
            default_width: args.viewport_width,
            zoom_factor: 1.0,
            y_zoom_factor: 1.0,
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
        "Invalid".to_string()
    } else {
        format!("{:.0}", x)
    }
}

impl eframe::App for LivePlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut first_x = f64::MAX;
            let mut last_x = f64::MIN;
            let mut has_data = false;

            {
                let data_lock = self.data.lock().unwrap();
                for buf in data_lock.iter() {
                    if let (Some(f), Some(l)) = (buf.first(), buf.last()) {
                        first_x = first_x.min(f[0]);
                        last_x = last_x.max(l[0]);
                        has_data = true;
                    }
                }
            }
            if !has_data {
                first_x = 0.0;
                last_x = 0.0;
            }

            let mut interaction_triggered = false;
            let layer_id = ui.layer_id();

            // --- 1. HEADER (X-Zoom Control) ---
            ui.horizontal(|ui| {
                ui.heading(&self.title);
                if self.stream_ended.load(Ordering::Relaxed) {
                    ui.colored_label(Color32::GOLD, "⚠️ Ended");
                } else if self.auto_follow {
                    ui.colored_label(Color32::LIGHT_BLUE, "• LIVE");
                }
                ui.separator();
                ui.label("X-Zoom:");

                let current_width = self.default_width / self.zoom_factor;
                let current_center_x = self.scroll_offset + (current_width / 2.0);

                let slider_w = (ui.available_width() - 100.0).max(50.0);
                let min_z =
                    (self.default_width / (last_x - first_x).max(self.default_width)).max(0.0001);
                let z_resp = ui.add_sized(
                    [slider_w, 20.0],
                    egui::Slider::new(&mut self.zoom_factor, min_z..=1000.0)
                        .show_value(false)
                        .logarithmic(true),
                );

                if z_resp.changed() {
                    self.auto_follow = false;
                    interaction_triggered = true;
                    let new_width = self.default_width / self.zoom_factor;
                    self.scroll_offset = current_center_x - (new_width / 2.0);
                }

                if ui.button("Reset").clicked() {
                    self.zoom_factor = 1.0;
                    self.y_zoom_factor = 1.0;
                    self.auto_follow = true;
                    interaction_triggered = true;
                }
            });

            let view_width = self.default_width / self.zoom_factor;
            if self.auto_follow {
                self.scroll_offset = (last_x - view_width).max(first_x);
            }

            // Clone data for closure
            let data_arc = self.data.clone();
            let is_ts = self.is_ts_mode;
            let include_y = self.include_y_values.clone();
            let labels = self.labels.clone();
            let colors = self.colors.clone();

            let mut scroll_offset = self.scroll_offset;
            let mut auto_follow = self.auto_follow;
            let mut zoom_factor = self.zoom_factor;
            let mut y_zoom_factor = self.y_zoom_factor;
            let def_w = self.default_width;

            // --- 2. MAIN BODY (Vertical Slider + Full Height Plot) ---
            // Using a specific layout to ensure the vertical slider and plot stretch
            let body_layout =
                egui::Layout::left_to_right(egui::Align::Min).with_cross_justify(true);

            ui.with_layout(body_layout, |ui| {
                // Vertical Slider
                ui.add_sized(
                    [20.0, ui.available_height()],
                    egui::Slider::new(&mut y_zoom_factor, 0.1..=1000.0)
                        .vertical()
                        .show_value(false)
                        .logarithmic(true),
                );

                if ui.input(|i| i.pointer.any_down())
                    && ui
                        .available_rect_before_wrap()
                        .contains(ui.input(|i| i.pointer.interact_pos().unwrap_or_default()))
                {
                    // Check if slider was touched
                }

                if y_zoom_factor != self.y_zoom_factor {
                    auto_follow = false;
                    interaction_triggered = true;
                }

                // Plot stretched to remaining width and height
                let num_series = labels.len();
                let plot = Plot::new("live_plot")
                    .height(ui.available_height())
                    .width(ui.available_width())
                    .auto_bounds([false, false].into())
                    .allow_zoom(true)
                    .allow_drag(true)
                    .show_x(false)
                    .show_y(false)
                    .label_formatter(|_, _| String::new())
                    .x_axis_formatter(move |mark, _| format_x_val(mark.value, is_ts))
                    .legend(Legend::default().position(self.legend_corner));

                let plot = include_y.iter().fold(plot, |p, &y| p.include_y(y));

                let plot_res = plot.show(ui, |plot_ui| {
                    let data_lock = data_arc.lock().unwrap();
                    let bounds = plot_ui.plot_bounds();

                    if plot_ui.pointer_coordinate_drag_delta().length() > 0.0
                        || plot_ui.ctx().input(|i| i.raw_scroll_delta.y).abs() > 0.0
                    {
                        auto_follow = false;
                    }

                    if auto_follow || interaction_triggered {
                        let mut min_y = f64::INFINITY;
                        let mut max_y = f64::NEG_INFINITY;
                        for buf in data_lock.iter() {
                            for p in buf.iter().filter(|p| {
                                p[0] >= scroll_offset && p[0] <= scroll_offset + view_width
                            }) {
                                min_y = min_y.min(p[1]);
                                max_y = max_y.max(p[1]);
                            }
                        }
                        for &y in &include_y {
                            min_y = min_y.min(y);
                            max_y = max_y.max(y);
                        }

                        let base_range_y = if min_y.is_infinite() {
                            (bounds.min()[1], bounds.max()[1])
                        } else {
                            let p = (max_y - min_y).max(0.1) * 0.05;
                            (min_y - p, max_y + p)
                        };

                        let center_y = (base_range_y.0 + base_range_y.1) / 2.0;
                        let half_h = ((base_range_y.1 - base_range_y.0) / 2.0) / y_zoom_factor;

                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [scroll_offset, center_y - half_h],
                            [scroll_offset + view_width, center_y + half_h],
                        ));
                    } else {
                        scroll_offset = bounds.min()[0];
                        if bounds.width() > 0.0 {
                            zoom_factor = def_w / bounds.width();
                        }

                        let mut b_min = f64::INFINITY;
                        let mut b_max = f64::NEG_INFINITY;
                        for buf in data_lock.iter() {
                            for p in buf
                                .iter()
                                .filter(|p| p[0] >= bounds.min()[0] && p[0] <= bounds.max()[0])
                            {
                                b_min = b_min.min(p[1]);
                                b_max = b_max.max(p[1]);
                            }
                        }
                        let b_h = (b_max - b_min).max(0.1);
                        if bounds.height() > 0.0 {
                            y_zoom_factor = (b_h / bounds.height()).max(0.001);
                        }
                    }

                    // Zero-copy parametric rendering
                    for (i, buffer) in data_lock.iter().enumerate() {
                        let series_len = buffer.len();
                        if series_len == 0 {
                            continue;
                        }
                        let points = PlotPoints::from_parametric_callback(
                            move |t| {
                                let idx = t.round() as usize;
                                if idx < series_len {
                                    (buffer[idx][0], buffer[idx][1])
                                } else {
                                    (0.0, 0.0)
                                }
                            },
                            0.0..=(series_len as f64 - 1.0),
                            series_len,
                        );
                        plot_ui.line(Line::new(points).name(&labels[i]).color(colors[i]));
                    }

                    // Hover Tooltip
                    if let Some(mouse_p) = plot_ui.pointer_coordinate() {
                        if let Some(ref_buf) = data_lock.get(0) {
                            let idx = ref_buf
                                .binary_search_by(|p| p[0].partial_cmp(&mouse_p.x).unwrap())
                                .unwrap_or_else(|e| e);
                            let mut best = None;
                            let mut best_dist = f64::INFINITY;
                            for i in (idx.saturating_sub(1))..(idx + 1).min(ref_buf.len()) {
                                for s_idx in 0..num_series {
                                    if let Some(val) = data_lock.get(s_idx).and_then(|b| b.get(i)) {
                                        let dx = val[0] - mouse_p.x;
                                        let dy = (val[1] - mouse_p.y)
                                            * (bounds.width() / bounds.height().max(0.1));
                                        let d = dx * dx + dy * dy;
                                        if d < best_dist {
                                            best_dist = d;
                                            best = Some((s_idx, val[0], val[1]));
                                        }
                                    }
                                }
                            }
                            if let Some((s_idx, x, y)) = best {
                                if best_dist < (bounds.width() * 0.015).powi(2) {
                                    let x_fmt = format_x_val(x, is_ts);
                                    let lbl = labels[s_idx].clone();
                                    egui::show_tooltip_at_pointer(
                                        plot_ui.ctx(),
                                        layer_id,
                                        egui::Id::new("plot_tt"),
                                        |ui: &mut egui::Ui| {
                                            ui.label(format!(
                                                "Series: {}\nVal: {:.4}\nX: {}",
                                                lbl, y, x_fmt
                                            ));
                                        },
                                    );
                                }
                            }
                        }
                    }
                });

                // Update state
                self.auto_follow = auto_follow;
                self.scroll_offset = scroll_offset;
                self.zoom_factor = zoom_factor;
                self.y_zoom_factor = y_zoom_factor;

                if plot_res.response.dragged() || interaction_triggered {
                    ui.ctx().request_repaint();
                }
            });
        });
    }
}

fn main() {
    let args = Args::parse();
    let is_ts = args.timestamp;
    let lbl_count = args.labels.as_ref().map_or(1, |l| l.len());
    let expected = if is_ts { lbl_count + 1 } else { lbl_count };
    let max_pts = args.max_points;

    let mut initial_buffers = Vec::with_capacity(lbl_count);
    for _ in 0..lbl_count {
        initial_buffers.push(Vec::with_capacity(max_pts));
    }
    let data_buffer = Arc::new(Mutex::new(initial_buffers));
    let data_buffer_thread = Arc::clone(&data_buffer);
    let stream_ended = Arc::new(AtomicBool::new(false));
    let stream_ended_thread = Arc::clone(&stream_ended);

    eframe::run_native(
        "Live Plotter",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 750.0]),
            ..Default::default()
        },
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                let mut seq_counter: u64 = 0;
                for (l_idx, line) in stdin.lock().lines().enumerate() {
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
                            eprintln!("FATAL: line {} cols.", l_idx + 1);
                            process::exit(1);
                        }
                        let (x, series) = if is_ts {
                            (vals[0], vals[1..].to_vec())
                        } else {
                            (seq_counter as f64, vals)
                        };
                        {
                            let mut buffers = data_buffer_thread.lock().unwrap();
                            for (i, val) in series.into_iter().enumerate() {
                                if i < buffers.len() {
                                    if buffers[i].len() >= max_pts {
                                        buffers[i].remove(0);
                                    }
                                    buffers[i].push([x, val]);
                                    if is_ts {
                                        let mut j = buffers[i].len() - 1;
                                        while j > 0 && buffers[i][j][0] < buffers[i][j - 1][0] {
                                            buffers[i].swap(j, j - 1);
                                            j -= 1;
                                        }
                                    }
                                }
                            }
                            seq_counter += 1;
                        }
                        ctx.request_repaint();
                    } else {
                        break;
                    }
                }
                stream_ended_thread.store(true, Ordering::Relaxed);
                ctx.request_repaint();
            });
            Ok(Box::new(LivePlotApp::new(args, data_buffer, stream_ended)))
        }),
    )
    .unwrap();
}
