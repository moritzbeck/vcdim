extern crate polygon;
extern crate vcdim;

#[cfg(feature = "print_info")]
use polygon::generate::Mode;
#[cfg(feature = "print_info")]
use vcdim::*;

#[cfg(feature = "print_info")]
fn main() {
    let mut args = std::env::args().skip(1);
    let n = if let Some(n) = args.next() {
        n.parse().expect("n must be an positive integer")
    } else { 20 };
    let count = if let Some(cnt) = args.next() {
        cnt.parse().expect("count must be an positive integer")
    } else { 1 };

    let gen_mode = Mode::QuickStarLike;
    let mut vcd = VcDim::with_random_polygon(n, gen_mode);

    for _ in 0..count {
        let _ = vcd.vc_dimension();
        vcd.randomize_polygon(gen_mode)
    }
}

#[cfg(not(feature = "print_info"))]
fn main() {
    println!("WARNING: This does nothing if the feature `print_info` is disabled.");
    println!("         Please enable this feature.");
    return;
}
