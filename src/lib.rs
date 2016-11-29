//#![warn(missing_docs)]
#![cfg_attr(feature = "bench", feature(test))]

#[cfg(all(feature = "bench", test))]
extern crate test;

extern crate polygon;
extern crate rand;

use polygon::*;
use std::io::{Read, Write};
use std::cell::RefCell;

#[derive(Debug)]
pub struct VcDim {
    /// The polygon of which the VC dimension is considered.
    pub polygon: Polygon,
    /// Visibility matrix of the polygon.
    /// The boolean `visible[i][j]` is true iff `points[i]` sees `points[j]`.
    pub visible: Vec<Vec<bool>>,
    // Holds an u8 giving the VC-dimension
    // and a Vec<usize> giving the indices of an maximal shattered subset.
    _vc_dimension_cache: RefCell<Option<(u8, Vec<usize>)>>
}
impl VcDim {
    pub fn new(p: Polygon) -> VcDim {
        assert!(p.size() >= 3);
        assert!(p.is_simple());
        let mut vis = Vec::with_capacity(p.size());
        for _ in 0..p.size() {
            vis.push(vec![true; p.size()]);
        }
        let mut v = VcDim {
            visible: vis,
            polygon: p,
            _vc_dimension_cache: RefCell::new(None)
        };
        v._calculate_visibility();
        v
    }
    pub fn with_random_polygon(n: usize, mode: generate::Mode) -> VcDim {
        assert!(n >= 3);
        let mut vis = Vec::with_capacity(n);
        for _ in 0..n {
            vis.push(vec![true; n]);
        }
        let mut v = VcDim {
            visible: vis,
            polygon: Polygon::from_points(&vec![Point::new_u(0, 0); n]),
            _vc_dimension_cache: RefCell::new(None)
        };
        v.randomize_polygon(mode);
        v
    }
    pub fn randomize_polygon(&mut self, mode: generate::Mode) {
        self.polygon.randomize(mode);

        // reset the cache
        self._vc_dimension_cache = RefCell::new(None);
        // The following lines are needed because of
        // an assumption by `_calculate_visibility`.
        for v in &mut self.visible.iter_mut() {
            for e in &mut v.iter_mut() {
                *e = true;
            }
        }
        self._calculate_visibility();
    }
    fn _calculate_visibility(&mut self) {
        // !!! Assumes that self.visible is intialised with 'true' set for each element !!!
        let points = self.polygon.points();
        for i in 0..points.len() {
            for j in i+2..points.len() {
                if i == 0 && j == points.len()-1 {
                    break; // they see each other
                }
                let p1 = points[i];
                let p2 = points[j];
                let mid = Point::new(
                    (p1.x + p2.x) / 2f64,
                    (p1.y + p2.y) / 2f64
                );
                if !self.polygon.contains(mid) {
                    self.visible[i][j] = false;
                    self.visible[j][i] = false;
                    continue; // they don't see each other
                }
                for l in self.polygon.edges() {
                    if p1 == l.from || p1 == l.to ||
                       p2 == l.from || p2 == l.to {
                            continue;
                    }
                    if l.intersects(&Line::new(p1, p2)) {
                        self.visible[i][j] = false;
                        self.visible[j][i] = false;
                        break; // they don't see each other
                    }
                }
            }
        }
    }
    fn _is_shattered(&self, p: &[usize]) -> bool {
        let mut visible_p = vec![false; 1 << p.len()]; // bitset for all subsets of p
        for i in 0..self.polygon.size() {
            let mut idx = 0;
            for j in 0..p.len() {
                if self.visible[i][p[j]] {
                    idx |= 1 << j;
                }
            }
            visible_p[idx] = true;
        }

        visible_p.into_iter().all(|b| b)
    }
    /// Determines if the given set of `Point`s is shattered.
    ///
    /// Returns false if one of the given `Point`s is not a vertex of `self.polygon`.
    pub fn is_shattered(&self, p: &[Point]) -> bool {
        let mut indices = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            if let Some(idx) = self.polygon.points().iter().position(|&x| x == p[i]) {
                indices.push(idx);
            } else {
                return false;
            }
        }
        self._is_shattered(&indices)
    }
    /// Returns the points of `self.polygon`.
    ///
    /// Convenience method equal to `self.polygon.points()`.
    pub fn points(&self) -> &[Point] {
        self.polygon.points()
    }
    fn _edge_tuples(&self, size: usize) -> SubsetIter {
        assert!(self.polygon.size() >= size);
        SubsetIter {
            size: size,
            max: self.polygon.size() - 1,
            state: SubSetIterState::New
        }
    }
    fn _edge_tuples_from_previous_set(&self, set: Vec<usize>) -> SubsetIter {
        assert!(self.polygon.size() >= set.len() + 1);
        let max = self.polygon.size() - 1;
        let len = set.len();
        if len == 0 || set[len-1] == len-1 {
            return SubsetIter {
                size: len + 1,
                max: max,
                state: SubSetIterState::New
            };
        }
        let mut set = set;
        set[len - 1] -= 1;
        for i in (0..len-1).rev() {
            if set[i] == set[i+1] {
                set[i] -= 1;
                set[i+1] = max + i + 1 - len; //max for this position
            }
        }
        set.push(max);
        SubsetIter {
            size: len + 1,
            max: max,
            state: SubSetIterState::Subset(set)
        }
    }
    fn _compute_vc_dimension_naive(&self) -> (u8, Vec<usize>) {
        if let Some((dim, ref subset)) = *self._vc_dimension_cache.borrow() {
            return (dim, subset.to_vec()); // clone subset
        }
        if self.polygon.size() <= 3 { return (0, Vec::new()); }
        let mut vc_dim = 0;
        let mut shatterable = true;
        let mut subset_size = 1;
        let mut shattered_subset = Vec::new();
        while shatterable {
            assert_eq!(vc_dim +1, subset_size);
            shatterable = false;
            if 1 << subset_size > self.polygon.size() {
                break; // more nodes needed to achieve this vc_dim
            }
            for subset in self._edge_tuples(subset_size) {
                if self._is_shattered(&subset[..]) {
                    shatterable = true;
                    vc_dim = subset_size;
                    shattered_subset = subset;
                    break;
                }
            }
            subset_size += 1;
        }
        let vc = (vc_dim as u8, shattered_subset.to_vec()); // copy shattered_subset
        let mut cache = self._vc_dimension_cache.borrow_mut(); // panics if the cache is currently borrowed (should not happen!)
        *cache = Some(vc);

        (vc_dim as u8, shattered_subset)
    }
    fn _compute_vc_dimension(&self) -> (u8, Vec<usize>) {
        if let Some((dim, ref subset)) = *self._vc_dimension_cache.borrow() {
            return (dim, subset.to_vec()); // clone subset
        }
        if self.polygon.size() <= 3 { return (0, Vec::new()); }
        let mut vc_dim = 0;
        let mut shatterable = true;
        let mut subset_size = 1;
        let mut shattered_subset = Vec::new();
        while shatterable {
            assert_eq!(vc_dim +1, subset_size);
            shatterable = false;
            if 1 << subset_size > self.polygon.size() {
                break; // more nodes needed to achieve this vc_dim
            }
            for subset in self._edge_tuples_from_previous_set(shattered_subset.to_vec()) { //copy shattered_subset
                if self._is_shattered(&subset[..]) {
                    shatterable = true;
                    vc_dim = subset_size;
                    shattered_subset = subset;
                    break;
                }
            }
            subset_size += 1;
        }
        let vc = (vc_dim as u8, shattered_subset.to_vec()); // copy shattered_subset
        let mut cache = self._vc_dimension_cache.borrow_mut(); // panics if the cache is currently borrowed (should not happen!)
        *cache = Some(vc);

        (vc_dim as u8, shattered_subset)
    }
    /// Returns the VC-dimension of `self.polygon`.
    ///
    /// Caches the result of the computation.
    /// If `self.vc_dimension` or `self.max_shattered_subset` was called before,
    /// just returns the cached value.
    pub fn vc_dimension(&self) -> u8 {
        let (vc_dim, _) = self._compute_vc_dimension();
        vc_dim
    }
    /// Returns a maximum shatttered subset of `self.polygon`.
    ///
    /// Caches the result of the computation.
    /// If `self.vc_dimension` or `self.max_shattered_subset` was called before,
    /// just returns the cached value.
    pub fn max_shattered_subset(&self) -> Vec<Point> {
        let (_, subset) = self._compute_vc_dimension();
        subset.into_iter().map(|i| self.polygon.points()[i]).collect()
    }
}
enum SubSetIterState {
    New,
    Subset(Vec<usize>),
    Stopped
}
struct SubsetIter {
    size: usize,
    max: usize,
    state: SubSetIterState
}
impl Iterator for SubsetIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.state {
            SubSetIterState::Stopped => return None,
            SubSetIterState::New => {
                let mut v = Vec::with_capacity(self.size);
                for i in 0..self.size {
                    v.push(i);
                }
                self.state = SubSetIterState::Subset(v);
                if let SubSetIterState::Subset(ref v) = self.state {
                    return Some(v.to_vec()); // copy v
                }
            },
            SubSetIterState::Subset(ref mut v) => {
                for i in (0..v.len()).rev() {
                    debug_assert!(v[i] <= self.max + i + 1 - v.len());
                    if v[i] == self.max + i + 1 - v.len() {
                        //println!("maxed: v[{}] = {}", i, v[i]);
                        continue;
                    } else {
                        v[i] += 1;
                        for j in i+1..v.len() {
                            v[j] = v[j-1] + 1;
                        }
                        return Some(v.to_vec()); // copy v
                    }
                }
            }
        }
        // The execution gets here only if we transition from Subset(v) to Stopped
        self.state = SubSetIterState::Stopped;
        None
    }
}

pub trait IpeExport {
    fn export_ipe<W: Write>(&self, w: W, scale: f64) -> std::io::Result<()>;
}
impl IpeExport for VcDim {
    fn export_ipe<W: Write>(&self, mut w: W, scale: f64) -> std::io::Result<()> {
        try!(write!(w, r#"<ipe version="70206" creator="libvcdim">
<ipestyle name="vc-poly">
<symbol name="vc-point" transformations="translations">
<path fill="blue">-1.8 -1.8 m 1.8 -1.8 l 1.8 1.8 l -1.8 1.8 l h</path>
</symbol>
<symbol name="vc-point(s)" transformations="translations">
<path fill="sym-stroke">-1.8 -1.8 m 1.8 -1.8 l 1.8 1.8 l -1.8 1.8 l h</path>
</symbol>
<color name="red" value="1 0 0"/>
<color name="green" value="0 1 0"/>
<color name="blue" value="0 0 1"/>
<color name="yellow" value="1 1 0"/>
<color name="orange" value="1 0.647 0"/>
<color name="purple" value="0.627 0.125 0.941"/>
<dashstyle name="normal" value="[]0"/>
<dashstyle name="dashed" value="[3 3]0"/>
</ipestyle>
<page>
<path>"#));
        let points = self.polygon.points();
        if points.len() > 0 {
            try!(write!(w, "{} {} m ", points[0].x * scale, points[0].y * scale));
        }
        for p in points.iter().skip(1) {
            try!(write!(w, "{} {} l ", p.x * scale, p.y * scale));
        }
        try!(write!(w, "h</path>\n"));
        for p in self.max_shattered_subset() {
            try!(write!(w, r#"<use name="vc-point" pos="{} {}"/>"#, p.x * scale, p.y * scale));
        }
        for (i, p) in points.iter().enumerate() {
            if self.max_shattered_subset().contains(p) {
                try!(write!(w, r#"<text pos="{} {}" size="6" stroke="blue" matrix="1 0 0 1 2 0" valign="center">{}</text>"#, p.x * scale, p.y * scale, i));
            } else {
                try!(write!(w, r#"<text pos="{} {}" size="3" valign="center">{}</text>"#, p.x * scale, p.y * scale, i));
            }
        }
        write!(w, "</page>\n</ipe>")
    }
}
#[derive(Debug)]
pub enum IpeImportError { //TODO?: doesn't impl Error for now
    IoError(std::io::Error),
    Malformed,
    SubsetNotShattered(VcDim)
}
impl From<std::io::Error> for IpeImportError {
    fn from(err: std::io::Error) -> Self {
        IpeImportError::IoError(err)
    }
}
pub trait IpeImport {
    fn import_ipe<R: Read>(r: R, scale: f64) -> Result<VcDim, IpeImportError>;
}
impl IpeImport for VcDim {
    fn import_ipe<R: Read>(mut r: R, scale: f64) -> Result<VcDim, IpeImportError> {
        // TODO: do parsing with regex.
        let mut file_contents = String::new();
        try!(r.read_to_string(&mut file_contents));
        let vcd;
        // TODO: rewrite using Option::and_then()
        if let Some(idx_start) = file_contents.find("<path>") { //TODO: make more robust by allowing attrs on path element
            let idx_start = idx_start + 6; // 6 == "<path>".len()
            if let Some(idx_end) = file_contents[idx_start..].find("</path>") {
                // all points are listed between indices idx_start and idx_end
                let points_str = &file_contents[idx_start..(idx_start+idx_end)].trim();
                assert_eq!(&points_str[points_str.len()-1..], "h");
                if let Some(idx) = points_str.rfind('l') { // chop off last 'l'
                    let points = points_str[..idx].split(|c| c == 'l' || c == 'm')
                        .map(|l| {
                            let split = l.trim().split(' ').collect::<Vec<&str>>();
                            let x = split[0].parse::<f64>().expect("Couldn't parse f64") * scale;
                            let y = split[1].parse::<f64>().expect("Couldn't parse f64") * scale;
                            Point::new(x,y)
                        })
                        .collect::<Vec<Point>>();
                    let polygon = Polygon::from_points(&points);
                    vcd = VcDim::new(polygon);
                } else {
                    return Err(IpeImportError::Malformed);
                }
            } else {
                return Err(IpeImportError::Malformed);
            }
        } else {
            return Err(IpeImportError::Malformed);
        }
        // import shattered subset if present
        let mut shattered_subset = vec![];
        let mut read_position = 0;
        while let Some(idx_start) = file_contents[read_position..].find(r#"<use name="vc-point" pos=""#) {
            let idx_start = read_position + idx_start + 26; // 26 == r#"<use name="vc-point" pos=""#.len()
            if let Some(idx_end) = file_contents[idx_start..].find(r#""/>"#) {
                // all points are listed between indices idx_start and idx_end
                let point_str = &file_contents[idx_start..(idx_start+idx_end)].trim();
                let coords = point_str.split(' ').collect::<Vec<&str>>();
                let x = coords[0].parse::<f64>().expect("Couldn't parse f64") * scale;
                let y = coords[1].parse::<f64>().expect("Couldn't parse f64") * scale;
                shattered_subset.push(Point::new(x, y));
                read_position = idx_start+idx_end;
            } else {
                return Err(IpeImportError::Malformed);
            }
        }
        if shattered_subset.len() > 0 {
            if !vcd.is_shattered(&shattered_subset) {
                return Err(IpeImportError::SubsetNotShattered(vcd));
            }
            let mut indices = Vec::with_capacity(shattered_subset.len());
            for i in 0..shattered_subset.len() {
                if let Some(idx) = vcd.polygon.points().iter().position(|&x| x == shattered_subset[i]) {
                    indices.push(idx);
                } else {
                    return Err(IpeImportError::Malformed);
                }
            }
            let vc = (shattered_subset.len() as u8, indices); // copy shattered_subset
            let mut cache = vcd._vc_dimension_cache.borrow_mut(); // panics if the cache is currently borrowed (should not happen!)
            *cache = Some(vc);
        }
        Ok(vcd)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use polygon::*;

    #[test]
    fn _calculate_visibility_works() {
//                8     6
//   polygon =    |\   /|
//                | \ / |
//          a-----9  7  5
//          |           |
//          |           |
//          |           |
//          |           |
//          |           |
//          b--0     3--4
//             |     |
//             |     |
//             1-----2
        let points = [Point::new_u(1,1), Point::new_u(1,0), Point::new_u(3, 0),
            Point::new_u(3,1), Point::new_u(4,1), Point::new_u(4,3),
            Point::new_u(4,4), Point::new_u(3,3), Point::new_u(2,4),
            Point::new_u(2,3), Point::new_u(0,3), Point::new_u(0,1)];
        let polygon = Polygon::from_points(&points);
        let v = VcDim::new(polygon);

        for i in 0..v.polygon.size() {
            assert!(v.visible[i][i], "Point doesn't see itself");
            assert!(v.visible[i][(i+1)%v.polygon.size()], "Point doesn't see its neighbor");
            assert!(v.visible[(i+1)%v.polygon.size()][i], "Point doesn't see its neighbor");
        }

        assert!( v.visible[ 0][ 2], "Point doesn't see a vertex it should");
        assert!( v.visible[ 2][ 0], "Point doesn't see a vertex it should");
        assert!( v.visible[ 0][ 3], "Point doesn't see a vertex it should");
        // TODO: the next line is up to debate, as the obstructing wall is parallel to the sight line
        assert!(!v.visible[ 0][ 4], "Point does see a vertex it shouldn't");
        assert!( v.visible[ 0][ 5], "Point doesn't see a vertex it should");
        // TODO: the next line is up to debate, as the obstructing wall is parallel to the sight line
        assert!(!v.visible[ 0][ 6], "Point does see a vertex it shouldn't");
        assert!( v.visible[ 0][ 7], "Point doesn't see a vertex it should");
        assert!(!v.visible[ 0][ 8], "Point does see a vertex it shouldn't");
        assert!(!v.visible[ 8][ 0], "Point does see a vertex it shouldn't");
        assert!( v.visible[ 0][ 9], "Point doesn't see a vertex it should");
        assert!( v.visible[ 0][10], "Point doesn't see a vertex it should");
        assert!( v.visible[ 1][ 3], "Point doesn't see a vertex it should");
        assert!(!v.visible[ 1][ 4], "Point does see a vertex it shouldn't");
        assert!( v.visible[ 1][ 5], "Point doesn't see a vertex it should");
        assert!( v.visible[ 1][ 6], "Point doesn't see a vertex it should");
        assert!( v.visible[ 1][ 7], "Point doesn't see a vertex it should");
        assert!(!v.visible[ 1][ 8], "Point does see a vertex it shouldn't");

        assert!(!v.visible[ 6][ 8], "Point does see a vertex it shouldn't");
        assert!(!v.visible[11][ 1], "Point does see a vertex it shouldn't");
    }
    #[test]
    fn _is_shattered_works() {
//                8     6
//   polygon =    |\   /|
//                | \ / |
//          a-----9  7  5
//          |           |
//          |           |
//          |           |
//          |           |
//          |           |
//          b--0     3--4
//             |     |
//             |     |
//             1-----2
        let points = [Point::new_u(1,1), Point::new_u(1,0), Point::new_u(3, 0),
            Point::new_u(3,1), Point::new_u(4,1), Point::new_u(4,3),
            Point::new_u(4,4), Point::new_u(3,3), Point::new_u(2,4),
            Point::new_u(2,3), Point::new_u(0,3), Point::new_u(0,1)];
        let polygon = Polygon::from_points(&points);
        let v = VcDim::new(polygon);

        assert!(v._is_shattered(&[1, 4]));
        assert!(v._is_shattered(&[2]));
        // TODO: add tests
    }
    #[test]
    fn _edge_tuples_from_previous_set_works() {
        for size in 10..20 {
            let v = VcDim::with_random_polygon(size); // TODO: deterministic polygon to save time (only size of polygon is needed)
            let m = size - 1; //max index
            assert_eq!(Some(vec![0]), v._edge_tuples_from_previous_set(vec![]).next());
            assert_eq!(Some(vec![0,1]), v._edge_tuples_from_previous_set(vec![0]).next());
            assert_eq!(Some(vec![1,2]), v._edge_tuples_from_previous_set(vec![1]).next());
            assert_eq!(Some(vec![m-1, m]), v._edge_tuples_from_previous_set(vec![m-1]).next());
            assert_eq!(None, v._edge_tuples_from_previous_set(vec![m]).next());
            assert_eq!(Some(vec![1,2,3]), v._edge_tuples_from_previous_set(vec![1,2]).next());
            assert_eq!(Some(vec![1,5,6]), v._edge_tuples_from_previous_set(vec![1,5]).next());
            assert_eq!(None, v._edge_tuples_from_previous_set(vec![m-2,m]).next());
            assert_eq!(Some(vec![1,5,6,7]), v._edge_tuples_from_previous_set(vec![1,5,6]).next());
            assert_eq!(Some(vec![2,3,4,5]), v._edge_tuples_from_previous_set(vec![1,m-1,m]).next());
            assert_eq!(Some(vec![2,3,4,5,6]), v._edge_tuples_from_previous_set(vec![1,m-2,m-1,m]).next());
            assert_eq!(Some(vec![2,3,4,5,6]), v._edge_tuples_from_previous_set(vec![1,m-3,m-2,m]).next());
            assert_eq!(Some(vec![1,m-3,m-2,m-1,m]), v._edge_tuples_from_previous_set(vec![1,m-3,m-2,m-1]).next());
        }
    }
    #[test]
    fn _compute_vc_dimension_fns_are_equal() {
        for size in 3..20 {
            for _ in 0..10 {
                let v = VcDim::with_random_polygon(size);
                let dim1 = v._compute_vc_dimension_naive();
                let dim2 = v._compute_vc_dimension();
                assert_eq!(dim1, dim2);
            }
        }
    }
}
#[cfg(all(feature = "bench", test))]
mod bench {
    use super::*;
    use test::Bencher;

    #[bench]
    fn _compute_vc_dimension_naive_10(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(10);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    fn _compute_vc_dimension_10(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(10);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_naive_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    fn _compute_vc_dimension_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_naive_30(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(30);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    fn _compute_vc_dimension_30(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(30);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_naive_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    fn _compute_vc_dimension_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _is_shattered_2_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20);
        b.iter(|| v._is_shattered(&[1, 4]));
    }
    #[bench]
    fn _is_shattered_2_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40);
        b.iter(|| v._is_shattered(&[1, 4]));
    }
    #[bench]
    fn _is_shattered_2_80(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(80);
        b.iter(|| v._is_shattered(&[1, 4]));
    }
    #[bench]
    fn _is_shattered_3_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20);
        b.iter(|| v._is_shattered(&[1, 4, 7]));
    }
    #[bench]
    fn _is_shattered_3_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40);
        b.iter(|| v._is_shattered(&[1, 4, 7]));
    }
    #[bench]
    fn _is_shattered_3_80(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(80);
        b.iter(|| v._is_shattered(&[1, 4, 7]));
    }
}
