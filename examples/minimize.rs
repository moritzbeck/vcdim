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

fn main() {
    let file_name = "vc6_monotone.ipe";
    let out_dir = "out";
    let vcd = VcDim::import_ipe(fs::File::open(file_name).expect(&format!("{} not found", file_name)), 1.).expect("File is malformed!");

    let max_shattered_subsets = vcd.all_max_shattered_subsets();
    println!("Found {} max subsets of size {}.", max_shattered_subsets.len(), max_shattered_subsets[0].len());
    for (i, subset) in max_shattered_subsets.iter().enumerate() {
        assert!(vcd.is_shattered(subset));
        let vcd_min = minimize_w_subset(&vcd.polygon, subset);
        if vcd_min.polygon.size() <= 35 {
            println!("Export");
            let point_c = vcd_min.polygon.size();
            vcd_min.export_ipe(fs::File::create(format!("{}/{}_{}.n{}.ipe", out_dir, file_name, i, point_c)).unwrap(), 1f64).unwrap();
        }
    }
}

