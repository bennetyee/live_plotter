use clap::Parser;
use eframe::egui;
use egui::Color32;
use egui_plot::{Corner, Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};
use std::thread;
use std::process;

#[derive(Parser, Debug)]
#[command(author, version, about = "Live multi-series time-series plotter")]
struct Args {
    /// Maximum number of data points to display in the sliding window per series
    #[arg(short, long, default_value_t = 1000)]
    max_points: usize,

    /// Y-axis values that should always be visible
    #[arg(long, num_args = 1..)]
    include_y: Vec<f64>,

    /// Labels for the data series. Number of labels sets expected columns.
    #[arg(short, long, num_args = 1..)]
    labels: Option<Vec<String>>,

    /// Hex colors for the lines (e.g., #ff0000 #00ff00).
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
                eprintln!(
                    "FATAL ERROR: Invalid --legend-pos '{}'.\nAccepted values: LeftTop, RightTop, LeftBottom, RightBottom",
                    args.legend_pos
                );
                process::exit(1);
            }
        };

        // Default palette for multi-series visualization
        let default_palette = vec![
            Color32::from_rgb(255, 85, 85), Color32::from_rgb(85, 255, 85),
            Color32::from_rgb(85, 85, 255), Color32::from_rgb(255, 255, 85),
            Color32::from_rgb(255, 85, 255), Color32::from_rgb(85, 255, 255),
            Color32::from_rgb(255, 170, 0), Color32::from_rgb(170, 0, 255),
            Color32::from_rgb(0, 255, 170), Color32::from_rgb(255, 0, 127),
            Color32::from_rgb(170, 255, 0), Color32::from_rgb(0, 170, 255),
            Color32::from_rgb(255, 215, 180), Color32::from_rgb(128, 128, 128),
            Color32::from_rgb(170, 110, 40), Color32::from_rgb(0, 128, 128),
            Color32::from_rgb(230, 190, 255), Color32::from_rgb(128, 0, 0),
            Color32::from_rgb(170, 255, 195), Color32::from_rgb(128, 128, 0),
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
            colors.push(default_palette[i % default_palette.len()]);
        }

        Self {
            data,
            include_y_values: args.include_y,
            labels,
            colors,
            title: args.title,
            legend_corner,
        }
    }
}

impl eframe::App for LivePlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data_lock = self.data.lock().unwrap();
            
            ui.horizontal(|ui| {
                ui.heading(&self.title);
                ui.weak("| Hover for info | Double-click to reset scale");
            });

            let num_series = self.labels.len();
            let mut plot = Plot::new("live_plot")
                .view_aspect(ui.available_width() / ui.available_height().max(1.0))
                .allow_zoom(true)
                .allow_drag(true)
                .label_formatter(|name, value| {
                    if name.is_empty() {
                        format!("Seq: {:.0}\nVal: {:.4}", value.x, value.y)
                    } else {
                        format!("Series: {}\nVal: {:.4}\nSeq: {:.0}", name, value.y, value.x)
                    }
                });

            if num_series > 1 {
                plot = plot.legend(Legend::default().position(self.legend_corner));
            }

            plot = self.include_y_values.iter().fold(plot, |p, &y| p.include_y(y));

            plot.show(ui, |plot_ui| {
                for i in 0..num_series {
                    let points: PlotPoints = data_lock
                        .iter()
                        .map(|p| [p.seq as f64, p.values[i]])
                        .collect();

                    let line = Line::new(points)
                        .name(&self.labels[i])
                        .color(self.colors[i]);
                    
                    plot_ui.line(line);
                }
            });
        });
        ctx.request_repaint();
    }
}

fn main() {
    let args = Args::parse();
    
    let expected_cols = args.labels.as_ref().map_or(1, |l| l.len());
    let data_buffer = Arc::new(Mutex::new(VecDeque::with_capacity(args.max_points)));
    let data_buffer_stdin = Arc::clone(&data_buffer);
    let max_points = args.max_points;

    thread::spawn(move || {
        let stdin = io::stdin();
        let mut sequence_counter: u64 = 0;
        
        for (line_idx, line) in stdin.lock().lines().enumerate() {
            let line_num = line_idx + 1;
            if let Ok(line_str) = line {
                let trimmed = line_str.trim();
                if trimmed.is_empty() { continue; }

                let values: Vec<f64> = trimmed
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse::<f64>().ok())
                    .collect();

                if values.len() != expected_cols {
                    eprintln!(
                        "FATAL ERROR: Input schema mismatch at line {}.\nExpected {} columns, found {}.",
                        line_num, expected_cols, values.len()
                    );
                    process::exit(1);
                }

                let mut buffer = data_buffer_stdin.lock().unwrap();
                buffer.push_back(DataPoint {
                    seq: sequence_counter,
                    values,
                });
                
                sequence_counter += 1;
                if buffer.len() > max_points {
                    buffer.pop_front();
                }
            } else {
                break;
            }
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 600.0])
            .with_title("Real-Time Multi-Series Plotter"),
        ..Default::default()
    };

    let result = eframe::run_native(
        "Live Plotter",
        options,
        Box::new(|_cc| Box::new(LivePlotApp::new(args, data_buffer))),
    );

    match result {
        Ok(_) => process::exit(0),
        Err(e) => {
            eprintln!("GUI Error: {}", e);
            process::exit(1);
        }
    }
}