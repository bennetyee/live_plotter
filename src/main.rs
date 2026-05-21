use clap::Parser;
use eframe::egui;
use egui::Color32;
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints};
use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::process;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Parser, Debug)]
#[command(author, version, about = "Optimized Live Multi-Series Plotter")]
struct Args {
    /// Maximum number of data points to display in the total buffer per series
    #[arg(short, long, default_value_t = 10000)]
    max_points: usize,

    /// Initial number of sequence points visible in the viewport
    #[arg(short, long, default_value_t = 500.0)]
    viewport_width: f64,

    /// Y-axis values that should always be visible (baseline/ceiling)
    #[arg(long, num_args = 1..)]
    include_y: Vec<f64>,

    /// Labels for the data series (Number of labels sets expected columns)
    #[arg(short, long, num_args = 1..)]
    labels: Option<Vec<String>>,

    /// Hex colors for the lines (e.g., #ff0000 #00ff00)
    #[arg(short, long, num_args = 1..)]
    colors: Option<Vec<String>>,

    /// The title displayed at the top of the graph
    #[arg(short, long, default_value = "Live Time-Series Feed")]
    title: String,

    /// Legend position: LeftTop, RightTop, LeftBottom, RightBottom
    #[arg(long, default_value = "LeftTop")]
    legend_pos: String,
}

struct DataPoint {
    seq: u64,
    values: Vec<f64>,
}

struct LivePlotApp {
    data: Arc<Mutex<VecDeque<DataPoint>>>,
    include_y_values: Vec<f64>,
    labels: Vec<String>,
    colors: Vec<Color32>,
    title: String,
    legend_corner: Corner,

    // Viewport State
    default_viewport_width: f64,
    max_buffer_size: usize,
    zoom_factor: f64,
    scroll_offset: f64,
    auto_follow: bool,
}

impl LivePlotApp {
    fn new(args: Args, data: Arc<Mutex<VecDeque<DataPoint>>>) -> Self {
        let labels = args.labels.unwrap_or_else(|| vec!["Series 1".to_string()]);
        let legend_corner = match args.legend_pos.to_lowercase().as_str() {
            "lefttop" => Corner::LeftTop,
            "righttop" => Corner::RightTop,
            "leftbottom" => Corner::LeftBottom,
            "rightbottom" => Corner::RightBottom,
            _ => {
                eprintln!("FATAL ERROR: Invalid --legend-pos '{}'.", args.legend_pos);
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
            include_y_values: args.include_y,
            labels,
            colors,
            title: args.title,
            legend_corner,
            default_viewport_width: args.viewport_width,
            max_buffer_size: args.max_points,
            zoom_factor: 1.0,
            scroll_offset: 0.0,
            auto_follow: true,
        }
    }
}

impl eframe::App for LivePlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data_lock = self.data.lock().unwrap();

            let (first_seq, last_seq) =
                if let (Some(f), Some(l)) = (data_lock.front(), data_lock.back()) {
                    (f.seq as f64, l.seq as f64)
                } else {
                    (0.0, 0.0)
                };

            ui.horizontal(|ui| {
                ui.heading(&self.title);
                if self.auto_follow {
                    ui.colored_label(Color32::from_rgb(100, 200, 255), "• LIVE");
                }
                ui.separator();
                ui.label("Zoom:");

                let min_zoom = self.default_viewport_width / self.max_buffer_size as f64;
                let slider_width = (ui.available_width() - 110.0).max(50.0);

                let zoom_slider = ui.add_sized(
                    [slider_width, ui.spacing().interact_size.y],
                    egui::Slider::new(&mut self.zoom_factor, min_zoom..=20.0).show_value(false),
                );

                if zoom_slider.changed() {
                    self.auto_follow = false;
                }

                if ui.button("Reset View").clicked() {
                    self.zoom_factor = 1.0;
                    self.auto_follow = true;
                }
            });

            let current_view_width = self.default_viewport_width / self.zoom_factor;
            if self.auto_follow {
                self.scroll_offset = (last_seq - current_view_width).max(first_seq);
            }

            let num_series = self.labels.len();
            let mut plot = Plot::new("live_plot")
                .view_aspect(ui.available_width() / (ui.available_height() - 95.0).max(1.0))
                .auto_bounds([false, false].into())
                .allow_zoom(true)
                .allow_drag(true)
                .label_formatter(|name, value| {
                    format!("Series: {}\nVal: {:.4}\nSeq: {:.0}", name, value.y, value.x)
                });

            for &y in &self.include_y_values {
                plot = plot.include_y(y);
            }

            if num_series > 1 {
                plot = plot.legend(Legend::default().position(self.legend_corner));
            }

            plot.show(ui, |plot_ui| {
                let bounds = plot_ui.plot_bounds();

                // Detection: use raw_scroll_delta for newer egui compatibility
                let interaction = plot_ui.pointer_coordinate_drag_delta().length() > 0.0
                    || plot_ui.ctx().input(|i| i.raw_scroll_delta.y).abs() > 0.0;

                if interaction {
                    self.auto_follow = false;
                }

                if self.auto_follow {
                    let mut min_v = f64::INFINITY;
                    let mut max_v = f64::NEG_INFINITY;
                    for dp in data_lock.iter() {
                        if (dp.seq as f64) >= self.scroll_offset {
                            for &v in &dp.values {
                                min_v = min_v.min(v);
                                max_v = max_v.max(v);
                            }
                        }
                    }
                    for &y in &self.include_y_values {
                        min_v = min_v.min(y);
                        max_v = max_v.max(y);
                    }

                    let y_range = if min_v.is_infinite() {
                        (0.0, 1.0)
                    } else {
                        let pad = (max_v - min_v).max(0.1) * 0.05;
                        (min_v - pad, max_v + pad)
                    };

                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [self.scroll_offset, y_range.0],
                        [self.scroll_offset + current_view_width, y_range.1],
                    ));
                } else {
                    // Sync state from manual panning
                    self.scroll_offset = bounds.min()[0];
                    if bounds.width() > 0.0 {
                        self.zoom_factor = self.default_viewport_width / bounds.width();
                    }
                }

                // Render lines from all data in buffer
                for i in 0..num_series {
                    let points: PlotPoints = data_lock
                        .iter()
                        .map(|p| [p.seq as f64, p.values[i]])
                        .collect();
                    plot_ui.line(
                        Line::new(points)
                            .name(&self.labels[i])
                            .color(self.colors[i]),
                    );
                }
            });

            ui.add_space(10.0);
            ui.label("History Scroll (Drag to far right for LIVE):");
            let scroll_max = (last_seq - current_view_width).max(first_seq);

            let full_width = ui.available_width();
            let scroll_bar = ui.add_sized(
                [full_width, ui.spacing().interact_size.y],
                egui::Slider::new(&mut self.scroll_offset, first_seq..=scroll_max)
                    .show_value(false),
            );

            if scroll_bar.changed() {
                self.auto_follow = self.scroll_offset >= (scroll_max - 0.01);
            }
        });

        // Repaint is requested only when data arrives in the thread below
    }
}

fn main() {
    let args = Args::parse();
    let expected_cols = args.labels.as_ref().map_or(1, |l| l.len());
    let max_points = args.max_points;

    let data_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(max_points)));
    let data_buffer_stdin = Arc::clone(&data_buffer);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Live Plotter",
        options,
        Box::new(move |cc| {
            let ctx = cc.egui_ctx.clone();
            thread::spawn(move || {
                let stdin = io::stdin();
                let mut sequence_counter: u64 = 0;
                for line in stdin.lock().lines() {
                    if let Ok(line_str) = line {
                        let trimmed = line_str.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let values: Vec<f64> = trimmed
                            .split(|c: char| c == ',' || c.is_whitespace())
                            .filter(|s| !s.is_empty())
                            .filter_map(|s| s.parse::<f64>().ok())
                            .collect();

                        if values.len() != expected_cols {
                            eprintln!(
                                "FATAL ERROR: expected {} columns, found {}.",
                                expected_cols,
                                values.len()
                            );
                            process::exit(1);
                        }

                        {
                            let mut buffer = data_buffer_stdin.lock().unwrap();
                            buffer.push_back(DataPoint {
                                seq: sequence_counter,
                                values,
                            });
                            sequence_counter += 1;
                            if buffer.len() > max_points {
                                buffer.pop_front();
                            }
                        }

                        // Only repaint when data actually arrives
                        ctx.request_repaint();
                    } else {
                        break;
                    }
                }
            });
            Ok(Box::new(LivePlotApp::new(args, data_buffer)))
        }),
    )
    .unwrap();
}
