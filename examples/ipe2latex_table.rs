extern crate vcdim;

use vcdim::*;

fn print_latex_table(vcd: &VcDim) {
    /*
    0 & (16, 16)  & $g_1$, $v_1$ \\
    1 & (48, 32)  & $v_{1,2}$ \\
    2 & (80, 16)  & $g_2$, $v_2$ \\
    3 & (112, 32) & $v_2$ \\
    4 & (144, 16) & $v_\emptyset$ \\
    5 & (80, 64)  & $v_{1,2}$ \\
    */
    let pts = vcd.points();
    let subset = vcd.max_shattered_subset();
    let g = vcd.points().into_iter().enumerate()
        .filter_map(|(i, p)| {
            if subset.contains(p) {
                Some(i)
            } else {
                None
            }
        }).collect::<Vec<usize>>();
    let mut g_counter = 1;
    for i in 0..vcd.polygon().size() {
        print!("{} & ({}, {}) & ", i, pts[i].x, pts[i].y);
        if g.contains(&i) {
            print!("$g_{}$, ", g_counter);
            g_counter += 1;
        }
        let mut seen_by = vec![];
        for j in 0..g.len() {
            if vcd.visible()[i][g[j]] {
                seen_by.push(j+1);
            }
        }
        if seen_by.is_empty() {
            println!("$v_\\emptyset$ \\\\");
        } else {
            let indices = seen_by.into_iter()
                .map( |i| format!("{}", i))
                .collect::<Vec<_>>()[..]
                .join(&",");
            println!("$v_{{{}}}$ \\\\", indices);
        }
    }
}

/// Prints a LaTeX table describing the given polygon (in an ipe-file).
/// Prints one row for every vertex consisting of the vertex number,
/// its coordinates and its visibility properties.
fn main() {
    let mut args = std::env::args().skip(1);
    let in_file = if let Some(arg) = args.next() {
        arg
    } else {
        println!("Please provide a file name!");
        return;
    };

    let vcd = VcDim::import_ipe(std::fs::File::open(in_file).expect("ipe file not found"), 1.).expect("File is malformed!");

    print_latex_table(&vcd);
}
