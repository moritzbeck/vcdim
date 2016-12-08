extern crate polygon;
extern crate vcdim;

use vcdim::*;
use std::fs;

fn main() {
    let vcd = VcDim::import_ipe(fs::File::open("vc6_monotone.ipe").expect("vc6_monotone.ipe not found"), 1.).expect("File is malformed!");

    let max_shattered_subsets = vcd.all_max_shattered_subsets();
    for subset in &max_shattered_subsets {
        assert!(vcd.is_shattered(subset));
    }
    println!("Found {} max subsets of size {}:\n{:?}", max_shattered_subsets.len(), max_shattered_subsets[0].len(), max_shattered_subsets);
}
