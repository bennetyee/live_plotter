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
#[command(author, version, about = "Live multi-series time plotter with strict validation")]
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
        
        let default_palette = vec![
            Color32::from_rgb(255, 85, 85),   // Red
            Color32::from_rgb(85, 255, 85),   // Green
            Color32::from_rgb(85, 85, 255),   // Blue
            Color32::from_rgb(255, 255, 85),  // Yellow
            Color32::from_rgb(255, 85, 255),  // Magenta
            Color32::from_rgb(85, 255, 255),  // Cyan
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
                ui.weak("| Double-click to reset scale");
            });

            let num_series = self.labels.len();
            let mut plot = Plot::new("live_plot")
                .view_aspect(ui.available_width() / ui.available_height().max(1.0))
                .allow_zoom(true)
                .allow_drag(true);

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
    
    // Determine expected columns: number of labels provided, or 1 if default
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

                // FATAL ERROR CHECK
                if values.len() != expected_cols {
                    eprintln!(
                        "FATAL ERROR: Input schema mismatch at line {}.\nExpected {} values, found {}.\nLine content: \"{}\"",
                        line_num, expected_cols, values.len(), trimmed
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
            .with_inner_size([1000.0, 500.0])
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