use clap::Parser;
use eframe::egui;
use egui::Color32;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};
use std::thread;
use std::process;

#[derive(Parser, Debug)]
#[command(author, version, about = "Live multi-series time plotter with hover support")]
struct Args {
    /// Maximum number of data points to display per line
    #[arg(short, long, default_value_t = 1000)]
    max_points: usize,

    /// Y-axis values that should always be visible
    #[arg(long, num_args = 1..)]
    include_y: Vec<f64>,

    /// Labels for the data series. The number of labels defines the expected column count.
    #[arg(short, long, num_args = 1..)]
    labels: Option<Vec<String>>,

    /// Hex colors for the lines (e.g., #ff0000 #00ff00).
    #[arg(short, long, num_args = 1..)]
    colors: Option<Vec<String>>,

    /// The title displayed at the top of the graph
    #[arg(short, long, default_value = "Live Time-Series Feed")]
    title: String,
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
}

impl LivePlotApp {
    fn new(args: Args, data: Arc<Mutex<VecDeque<DataPoint>>>) -> Self {
        let labels = args.labels.unwrap_or_else(|| vec!["Series 1".to_string()]);
        
        // Expanded 20-color qualitative palette for high contrast
        let default_palette = vec![
            Color32::from_rgb(255, 85, 85),   // Red
            Color32::from_rgb(85, 255, 85),   // Green
            Color32::from_rgb(85, 85, 255),   // Blue
            Color32::from_rgb(255, 255, 85),  // Yellow
            Color32::from_rgb(255, 85, 255),  // Magenta
            Color32::from_rgb(85, 255, 255),  // Cyan
            Color32::from_rgb(255, 170, 0),   // Orange
            Color32::from_rgb(170, 0, 255),   // Purple
            Color32::from_rgb(0, 255, 170),   // Teal
            Color32::from_rgb(255, 0, 127),   // Rose
            Color32::from_rgb(170, 255, 0),   // Lime
            Color32::from_rgb(0, 170, 255),   // Azure
            Color32::from_rgb(255, 215, 180), // Apricot
            Color32::from_rgb(128, 128, 128), // Grey
            Color32::from_rgb(170, 110, 40),  // Brown
            Color32::from_rgb(0, 128, 128),   // Dark Teal
            Color32::from_rgb(230, 190, 255), // Lavender
            Color32::from_rgb(128, 0, 0),     // Maroon
            Color32::from_rgb(170, 255, 195), // Mint
            Color32::from_rgb(128, 128, 0),   // Olive
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
        }
    }
}

impl eframe::App for LivePlotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let data_lock = self.data.lock().unwrap();
            
            ui.horizontal(|ui| {
                ui.heading(&self.title);
                ui.weak("| Hover line for details | Double-click to reset");
            });

            let num_series = self.labels.len();
            let mut plot = Plot::new("live_plot")
                .view_aspect(ui.available_width() / ui.available_height().max(1.0))
                .allow_zoom(true)
                .allow_drag(true)
                // Custom tooltip formatter: Shows "Label: Value (Seq: X)"
                .label_formatter(|name, value| {
                    if name.is_empty() {
                        format!("Seq: {:.0}\nVal: {:.4}", value.x, value.y)
                    } else {
                        format!("Series: {}\nVal: {:.4}\nSeq: {:.0}", name, value.y, value.x)
                    }
                });

            if num_series > 1 {
                plot = plot.legend(Legend::default());
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
                        "FATAL ERROR: Input schema mismatch at line {}.\nExpected {} values, found {}.",
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
            .with_title("Live Multi-Plotter"),
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