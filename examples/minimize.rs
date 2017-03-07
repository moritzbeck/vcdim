extern crate polygon;
extern crate vcdim;
extern crate rand;

use polygon::*;
use vcdim::*;
use std::fs;
use rand::distributions::{IndependentSample, Range};

fn minimize_w_subset(polygon: &Polygon, sh_set: &[Point]) -> VcDim {
    let mut vcd = VcDim::new(polygon.clone());
    let mut points = vcd.points().to_vec();
    let mut polygon;

    assert!(vcd.is_shattered(sh_set));
    assert!(vcd.polygon.is_simple());

    let mut rng = rand::thread_rng();
    let mut minimized = false;
    let mut removed_c = 0;
    while !minimized {
        if points.len() == 3 { break; }
        minimized = true;
        let range = Range::new(0, points.len());
        let rand_offset = range.ind_sample(&mut rng);
        for i in 0..points.len() { // go around the polygon
            let idx = (i + rand_offset) % points.len();
            let pt = points.remove(idx);
            polygon = Polygon::from_points(&points);
            if !polygon.is_simple() {
                points.insert(idx, pt);
                continue;
            }
            vcd = VcDim::new(polygon); // this is quite un-performant (I guess) as this recomputes the entire visibility matrix.
            if vcd.is_shattered(&sh_set) {
                minimized = false;
                removed_c += 1;
                break;
            } else {
                points.insert(idx, pt);
            }
        }
    }

    vcd = VcDim::new(Polygon::from_points(&points));
    println!("Removed {} vertices. New size: {}", removed_c, vcd.polygon.size());
    vcd
}
/// Returns if this polygon is small enough to be considered interesting.
fn is_sufficiently_small(vcd: &VcDim) -> bool {
    let vc_dim = vcd.vc_dimension();
    let size = vcd.polygon.size();
    match vc_dim {
        0 | 1 => false,  // we don't want those boring polygons
        2 => size <=  6, // size ==  6 is the minimum
        3 => size <=  8, // size ==  8 is the minimum
        4 => size <= 16, // size == 16 is the minimum
        5 => size <= 35, // size == 34 for the currently known smallest polygon
        6 => size <= 78, // size == 78 for the currently known smallest polygon; we export the original anyways
        _ => true        // should not be possible
    }
}
/// Minimizes a single given polygon.
///
/// First it computes every shattered subset of maximal size
/// (i.e. with `d` elements where `d` is the VC-Dimension).
/// Then for every of these subsets it removes vertices
/// such that this subset remains shattered.
fn main() {
    let mut args = std::env::args().skip(1);
    let file_name = &if let Some(arg) = args.next() {
        arg
    } else {
        println!("Please provide a file name!");
        return;
    };
    let out_dir = "out";
    let vcd = VcDim::import_ipe(fs::File::open(file_name).expect(&format!("{} not found", file_name)), 1.).expect("File is malformed!");

    let max_shattered_subsets = vcd.all_max_shattered_subsets();
    println!("Found {} max subsets of size {}.", max_shattered_subsets.len(), max_shattered_subsets[0].len());
    for (i, subset) in max_shattered_subsets.iter().enumerate() {
        assert!(vcd.is_shattered(subset));
        let vcd_min = minimize_w_subset(&vcd.polygon, subset);
        if is_sufficiently_small(&vcd_min) {
            println!("Export");
            let point_c = vcd_min.polygon.size();
            vcd_min.export_ipe(fs::File::create(format!("{}/{}_{}.n{}.ipe", out_dir, file_name, i, point_c)).unwrap(), 1f64).unwrap();
        }
    }
}

