extern crate vcdim;

use vcdim::*;

fn print_visibility(vcd: &VcDim) {
    let subset = vcd.max_shattered_subset();
    let p = vcd.points().into_iter().enumerate()
        .filter_map(|(i, p)| {
            if subset.contains(p) {
                Some(i)
            } else {
                None
            }
        }).collect::<Vec<usize>>();
    for i in 0..vcd.polygon.size() {
        for j in 0..p.len() {
            if vcd.visible[i][p[j]] {
                print!("{}", j);
            }
        }
        println!(":{}", i);
    }
}

/// Displays for each vertex of the given polygon
/// which of the VC-points it can see.
fn main() {
    let mut args = std::env::args().skip(1);
    let in_file = if let Some(arg) = args.next() {
        arg
    } else {
        println!("Please provide a file name!");
        return;
    };

    let vcd = VcDim::import_ipe(std::fs::File::open(in_file).expect("ipe file not found"), 1.).expect("File is malformed!");

    print_visibility(&vcd);
}
