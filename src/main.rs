use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis, IxDyn};
use ort::{execution_providers::CUDAExecutionProvider, inputs, session::Session};
use rocket::{form::Form, fs::TempFile, response::content};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant, vec};

#[macro_use]
extern crate rocket;

// Main function that defines
// a web service endpoints a starts
// the web service
#[rocket::main]
async fn main() {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()
        .unwrap();

    let model_session = Session::builder()
        .unwrap()
        .commit_from_file("yolov8m-pose.onnx")
        .unwrap();

    rocket::build()
        .manage(model_session)
        .mount("/", routes![index])
        .mount("/detect", routes![detect])
        .launch()
        .await
        .unwrap();
}

// Site main page handler function.
// Returns Content of index.html file
#[get("/")]
fn index() -> content::RawHtml<String> {
    content::RawHtml(std::fs::read_to_string("index.html").unwrap())
}

// Handler of /detect POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through YOLOv8 object detection network and returns and array
// of bounding boxes.
// Returns a JSON array of objects bounding boxes in format [(x1,y1,x2,y2,object_type,probability),..]
#[post("/", data = "<file>")]
fn detect(file: Form<TempFile<'_>>, model_session: &rocket::State<Session>) -> String {
    let buf = std::fs::read(file.path().unwrap_or(Path::new(""))).unwrap_or(vec![]);
    let boxes: Vec<BBox> = detect_objects_on_image(buf, &model_session);
    return serde_json::to_string(&boxes).unwrap_or_default();
}

// Defining a type for boxes
type TKeyPoint = (f32, f32, f32);

#[derive(Serialize, Deserialize, Clone)]
struct BBox {
    top_left: (f32, f32),
    bottom_right: (f32, f32),
    confidence: f32,
    keypoints: [TKeyPoint; 17],
}

// Function receives an image,
// passes it through YOLOv8 neural network
// and returns an array of detected objects
// and their bounding boxes
// Returns Array of bounding boxes in format [(x1,y1,x2,y2,object_type,probability),..]
fn detect_objects_on_image(buf: Vec<u8>, model_session: &Session) -> Vec<BBox> {
    let start_prepare = Instant::now();

    let (input, img_width, img_height) = prepare_input(buf);

    let prepare_duration = start_prepare.elapsed();
    println!("Time to prepare input: {:?}", prepare_duration);

    let start_inference = Instant::now();

    let output = run_model(input, &model_session);

    let inference_duration = start_inference.elapsed();
    println!("Inference time: {:?}", inference_duration);

    let start_process = Instant::now();

    let processed_output = process_output(output, img_width, img_height);

    let process_duration = start_process.elapsed();
    println!("Process time: {:?}", process_duration);

    return processed_output;
}

// Function used to convert input image to tensor,
// required as an input to YOLOv8 object detection
// network.
// Returns the input tensor, original image width and height
fn prepare_input(buf: Vec<u8>) -> (Array<f32, IxDyn>, u32, u32) {
    let start_load = Instant::now();

    let img = image::load_from_memory(&buf).unwrap();

    let load_duration = start_load.elapsed();
    println!("Time to load: {:?}", load_duration);

    let (img_width, img_height) = (img.width(), img.height());

    let start_resize = Instant::now();

    let img = img.resize_exact(640, 640, FilterType::CatmullRom);

    let resize_duration = start_resize.elapsed();
    println!("Time to resize: {:?}", resize_duration);

    let mut input = Array::zeros((1, 3, 640, 640)).into_dyn();

    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    return (input, img_width, img_height);
}

// Function used to pass provided input tensor to
// YOLOv8 neural network and return result
// Returns raw output of YOLOv8 network
fn run_model(input: Array<f32, IxDyn>, model_session: &Session) -> Array<f32, IxDyn> {
    let outputs = model_session
        .run(inputs!["images" => input.view()].unwrap())
        .expect("Something went wrong");

    let output: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> =
        outputs["output0"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .t()
            .into_owned();

    return output;
}

// Function used to convert RAW output from YOLOv8 to an array
// of detected objects. Each object contain the bounding box of
// this object, the type of object and the probability
// Returns array of detected objects in a format [(x1,y1,x2,y2,object_type,probability),..]
fn process_output(output: Array<f32, IxDyn>, img_width: u32, img_height: u32) -> Vec<BBox> {
    let mut boxes = Vec::new();
    let output_2d = output.slice(s![.., .., 0]);

    for row in output_2d.axis_iter(Axis(0)) {
        // let row: Vec<_> = row.iter().map(|x| *x).collect();
        let row: [f32; 56] = match row.iter().cloned().collect::<Vec<f32>>().try_into() {
            Ok(array) => array,
            Err(_) => continue, // Handle the case where the conversion fails
        };
        if row.len() < 56 {
            continue;
        }

        let confidence = row[4];
        if confidence < 0.5 {
            continue;
        }

        let xc = row[0] / 640.0 * (img_width as f32);
        let yc = row[1] / 640.0 * (img_height as f32);
        let w = row[2] / 640.0 * (img_width as f32);
        let h = row[3] / 640.0 * (img_height as f32);
        let x1 = xc - w / 2.0;
        let x2 = xc + w / 2.0;
        let y1 = yc - h / 2.0;
        let y2 = yc + h / 2.0;

        let mut keypoints: [TKeyPoint; 17] = [(0.0, 0.0, 0.0); 17]; // Preallocate array
        for i in 0..17 {
            let kx = (row[5 + i * 3] / 640.0) * (img_width as f32);
            let ky = (row[6 + i * 3] / 640.0) * (img_height as f32);
            let kc = row[7 + i * 3];
            keypoints[i] = (kx, ky, kc);
        }

        let b_box = BBox {
            top_left: (x1, y1),
            bottom_right: (x2, y2),
            confidence,
            keypoints,
        };

        boxes.push(b_box);
    }

    boxes.sort_by(|b1, b2| b2.confidence.total_cmp(&b1.confidence));

    let mut results = Vec::new();

    while let Some(best_box) = boxes.first().cloned() {
        results.push(best_box.clone());
        boxes.retain(|bx| iou(&best_box, bx) < 0.7); // Retains only non-overlapping boxes
    }

    results
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
// Returns Intersection over union ratio as a float number
fn iou(b1: &BBox, b2: &BBox) -> f32 {
    return intersection(b1, b2) / union(b1, b2);
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
fn union(b1: &BBox, b2: &BBox) -> f32 {
    let b1_area = (b1.bottom_right.0 - b1.top_left.0) * (b1.bottom_right.1 - b1.top_left.1);
    let b2_area = (b2.bottom_right.0 - b2.top_left.0) * (b2.bottom_right.1 - b2.top_left.1);
    return b1_area + b2_area - intersection(b1, b2);
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
fn intersection(b1: &BBox, b2: &BBox) -> f32 {
    let x1 = b1.top_left.0.max(b2.top_left.0);
    let x2 = b1.bottom_right.0.max(b2.bottom_right.0);
    let y1 = b1.top_left.1.max(b2.top_left.1);
    let y2 = b1.bottom_right.1.max(b2.bottom_right.1);

    return (x2 - x1) * (y2 - y1);
}
