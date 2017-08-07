extern crate polygon;
extern crate vcdim;
extern crate rand;

use polygon::*;
use polygon::generate::Mode;
use vcdim::*;
use rand::distributions::{IndependentSample, Range};
use std::fs::File;
use std::io::Write;

fn minimize_w_subset(polygon: &Polygon, sh_set: &[Point]) -> VcDim {
    let mut vcd = VcDim::new(polygon.clone());
    let mut points = vcd.points().to_vec();
    let mut polygon;

    assert!(vcd.is_shattered(sh_set));
    assert!(vcd.polygon().is_simple());

    let mut rng = rand::thread_rng();
    let mut minimized = false;
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
                break;
            } else {
                points.insert(idx, pt);
            }
        }
    }

    vcd = VcDim::new(Polygon::from_points(&points));
    vcd
}
fn is_sufficiently_small(vcd: &VcDim) -> bool {
    let vc_dim = vcd.vc_dimension();
    let size = vcd.polygon().size();
    match vc_dim {
        0 | 1 => false,  // we don't want those boring polygons
        2 => size <=  6, // size ==  6 is the minimum
        3 => size <=  8, // size ==  8 is the minimum
        4 => size <= 16, // size == 16 is the minimum
        5 => size <= 32, // size == 32 is the minimum
        6 => size <= 78, // size == 78 for the currently known smallest polygon; we export the original anyways
        _ => true        // should not be possible
    }
}

/// Generates random polygons and minimizes them.
///
/// For each generated polygon:
/// First it computes some shattered subsets of maximal size
/// (i.e. with `d` elements where `d` is the VC-Dimension).
/// Then for every of these subsets it removes vertices
/// such that this subset remains shattered.
fn main() {
    let mut args = std::env::args().skip(1);
    let n = if let Some(n) = args.next() {
        n.parse().expect("n must be an positive integer")
    } else { 20 };
    let count = if let Some(cnt) = args.next() {
        cnt.parse().expect("count must be an positive integer")
    } else { 10 };
    let default_gen_mode = Mode::QuickStarLike;
    let gen_mode = if let Some(m) = args.next() {
        if m.starts_with("--mode=") {
            match &m[7..] {
                "2opt" => Mode::TwoOptLike,
                "quickstar" => Mode::QuickStarLike,
                "xmonotone" => Mode::Xmonotone,
                "monotone" => Mode::Xmonotone,
                "xmonotonegap" => Mode::XmonotoneGap,
                "monotonegap" => Mode::XmonotoneGap,
                _ => {
                    println!("Generation mode not recognised: {}\nPossible Values: 2opt, quickstar, xmonotone", &m[7..]);
                    return;
                }
            }
        } else {
            println!("Unrecognised argument: {}", m);
            return;
        }
    } else { default_gen_mode };

    let out_dir = "out";

    println!("Visibility VC-dimension of Polygons\n===================================\n\nCreating {} Polygons with {} vertices", count, n);
    println!("Mode: {:?}", gen_mode);

    let mut stdout = std::io::stdout();

    let mut vcd = VcDim::with_random_polygon(n, gen_mode);

    for i in 0..count {
        let vc_dim = vcd.vc_dimension();
        //vcd.export_ipe(File::create(format!("{}/d{}-n{}_{:03}___.ipe", out_dir, vc_dim, n, i)).unwrap(), 1f64).unwrap();
        let max_shattered_subsets;
        if vc_dim >= 5 {
            vcd.export_ipe(File::create(format!("{}/vc{}-n{}_{:03}.ipe", out_dir, vc_dim, n, i)).unwrap(), 1f64).unwrap();
            max_shattered_subsets = vcd.all_max_shattered_subsets();
            print!("{}", vc_dim);
        } else if vc_dim == 4 {
            max_shattered_subsets = vec![]; // don't try to minmize these polygons
            //max_shattered_subsets = vcd.all_max_shattered_subsets();
            print!("{}", vc_dim);
        } else /*if vc_dim == 3 {
            max_shattered_subsets = vcd.first_n_max_shattered_subsets(10);
            print!("{}", vc_dim);
        } else*/ {
            max_shattered_subsets = vec![]; // don't try to minmize these polygons
            print!("{}", vc_dim);
        }
        for (j, subset) in max_shattered_subsets.iter().enumerate() {
            assert!(vcd.is_shattered(subset));
            let vcd = minimize_w_subset(vcd.polygon(), subset);
            if is_sufficiently_small(&vcd) {
                let n = vcd.polygon().size();
                vcd.export_ipe(File::create(format!("{}/vc{}-n{}_{:03}s{}.ipe", out_dir, vc_dim, n, i, j)).unwrap(), 1f64).unwrap();
            }
        }
        stdout.flush().expect("Couldn't flush stdout");

        // generate new polygon
        if i != count-1 {
            vcd.randomize_polygon(gen_mode);
        }
    }
    println!("");
}

