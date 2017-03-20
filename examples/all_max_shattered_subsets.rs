extern crate polygon;
extern crate vcdim;

use vcdim::*;
use std::fs;

/// Reads in an ipe-file and prints all shattered subsets of maximal size
/// (i.e. with `d` elements where `d` is the VC-Dimension) of the described polygon.
fn main() {
    let mut args = std::env::args().skip(1);
    let file_name = &if let Some(arg) = args.next() {
        arg
    } else {
        println!("Please provide a file name!");
        return;
    };

    let vcd = VcDim::import_ipe(fs::File::open(file_name).expect(&format!("{} not found", file_name)), 1.).expect("File is malformed!");

    let max_shattered_subsets = vcd.all_max_shattered_subsets();
    for subset in &max_shattered_subsets {
        assert!(vcd.is_shattered(subset));
    }
    println!("Found {} max subsets of size {}:\n{:#?}", max_shattered_subsets.len(), max_shattered_subsets[0].len(), max_shattered_subsets);
}
