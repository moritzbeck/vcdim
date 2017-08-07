extern crate polygon;
extern crate vcdim;

use vcdim::*;
use std::fs;

/// Checks whether given polygons are x-monotone.
///
/// Reads every ipe-file in a given directory.
/// If it cannot be read/parsed 'X' is printed,
/// if the contained polygon is x-monotone 'M' is printed
/// otherwise '.' is printed.
///
/// Also prints out the path of one x-monotone polygon
/// and of one non-x-monotone polygon if present.
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
                    let vcd;
                    match fs::File::open(p.clone()).map_err(|e| e.into()).and_then(|f| { VcDim::import_ipe(f, 1.) }) {
                        Ok(v) => vcd = v,
                        Err(_) => {
                            print!("X");
                            continue; // skip this file
                        }
                    }
                    if vcd.polygon().is_x_monotone() {
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
