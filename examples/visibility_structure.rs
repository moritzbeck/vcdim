extern crate vcdim;

use vcdim::*;
use std::fs;
use std::collections::HashMap;

fn visibility_structure(vcd: &VcDim) -> String {
    let pts = vcd.points();
    let sh = vcd.max_shattered_subset();
    vcd.visible.iter().enumerate().map(|(i,v)| {
        let c = v.iter().enumerate()
            .filter(|&(j, sees_i)| { sh.contains(&pts[j]) && *sees_i })
            .count();
        if sh.contains(&pts[i]) {
            (96 + c as u8) as char // 1 -> 'a', 2 -> 'b', etc.
        } else {
            (48 + c as u8) as char // 0 -> '0', 1 -> '1', etc.
        }
    }).collect::<String>()
}
/// Normalizes a `String` representing the visibility structure of a polygon.
///
/// The result is a cyclic permutation of `str` or of its reverse,
/// starts with a '0' and is the lexicographical greater one of the two possibilities.
fn normalize(str: &str) -> String {
    let pos_of_0 = str.bytes().position(|c| { c == b'0' })
        .expect("str is representation of a shattered set, so there must be one point seeing 0 points.");
    let (p1, p2) = str.split_at(pos_of_0);
    let str1 = format!("{}{}", p2, p1); // concat p2 and p1
    let (p1, p2) = str.split_at(pos_of_0 + 1);
    let str2 = format!("{}{}", p2, p1).chars().rev().collect::<String>(); // concat p2 and p1 and reverse the String
    std::cmp::max(str1, str2)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut sort = false;
    let mut in_dir = "in".into();
    while let Some(arg) = args.next() {
        if arg == "--sort" {
            sort = true;
        } else {
            in_dir = arg;
        }
    }
    let sort = sort; //make immutable
    let in_dir = in_dir; //make immutable

    let mut ipe_files = Vec::new();
    if let Ok(entries) = fs::read_dir(in_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
            if let Ok(file_type) = entry.file_type() {
            if file_type.is_file() {
                if entry.path().extension() == Some(std::ffi::OsStr::new("ipe")) {
                    ipe_files.push(entry.path());
                }
            }}}
        }
    }
    let mut vis_structures = HashMap::new();
    for path in ipe_files {
        let vcd = VcDim::import_ipe(fs::File::open(path).expect("file not found"), 1f64).expect("File is malformed!");
        let vis_str = normalize(&visibility_structure(&vcd));
        if !sort {
            println!("{}", vis_str);
        } else {
            let counter = vis_structures.entry(vis_str).or_insert(0);
            *counter += 1;
        }
    }
    if !sort { return; }
    let mut vis_counts: Vec<_> = vis_structures.iter().collect();
    vis_counts.sort_by(|a, b| b.1.cmp(a.1));

    for v in vis_counts {
        println!("{}: {}", v.0, v.1);
    }
}

