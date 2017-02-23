extern crate polygon;
extern crate vcdim;

use vcdim::*;
use std::fs;

fn main() {
    let mut args = std::env::args().skip(1);
    let in_dir = &if let Some(arg) = args.next() {
        arg
    } else {
        println!("Please provide a input directory!");
        return;
    };

    let mut monotone_example = None;
    let mut non_monotone_example = None;
    if let Ok(entries) = fs::read_dir(in_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
            if let Ok(file_type) = entry.file_type() {
            if file_type.is_file() {
                if entry.path().extension() == Some(std::ffi::OsStr::new("ipe")) {
                    // Import the file and determine if its polygon is monotone.
                    let p = entry.path();
                    let vcd = VcDim::import_ipe(fs::File::open(p.clone()).expect(&format!("{} not found", p.display())), 1.).expect("File is malformed!");
                    if vcd.polygon.is_x_monotone() {
                        print!("M");
                        if monotone_example.is_none() {
                            monotone_example = Some(p);
                        }
                    } else {
                        print!(".");
                        if non_monotone_example.is_none() {
                            non_monotone_example = Some(p);
                        }
                    }
                }
            }}}
        }
    }
    println!("\n");

    if let Some(f) = monotone_example {
        println!("This file contains a x-monotone polygon: {}", f.display())
    } else {
        println!("Found no x-monotone polygon.");
    }
    if let Some(f) = non_monotone_example {
        println!("This file contains a polygon that is not x-monotone: {}", f.display())
    } else {
        println!("All files contain x-monotone polygons.");
    }
}
