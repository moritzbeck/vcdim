extern crate polygon;
extern crate rand;

use polygon::*;
use std::io::{Read, Write};
use std::cell::RefCell;
use rand::distributions::{IndependentSample, Range};

#[derive(Debug)]
pub struct VcDim {
    polygon: Polygon,
    visible: Vec<Vec<bool>>,
    _vc_dimension_cache: RefCell<Option<(u8, Vec<usize>)>>
}
impl VcDim {
    pub fn new(p: Polygon) -> VcDim {
        assert!(p.size() >= 3);
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
    pub fn with_random_polygon(n: usize) -> VcDim {
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
        v.randomize_polygon();
        v
    }
    pub fn randomize_polygon(&mut self) {
        let mut rng = rand::thread_rng();
        let range = Range::new(1.0, 500.0);
        let len = self.polygon.size();
        {
            let points = self.polygon.points_mut();
            for i in 0..len {
                let rand_x = range.ind_sample(&mut rng);
                let rand_y = range.ind_sample(&mut rng);
                points[i] = Point::new(rand_x, rand_y);
            }
        }
        let d = self._disentangle_polygon();
        if d.is_none() {
            self.randomize_polygon(); // try again
            // TODO?: remove recursion
        }
        self._vc_dimension_cache = RefCell::new(None);
        for v in &mut self.visible.iter_mut() {
            for e in &mut v.iter_mut() {
                *e = true;
            }
        }
        self._calculate_visibility();
    }
    fn _disentangle_polygon(&mut self) -> Option<u32> {
        // Ω(n^2)
        let len = self.polygon.size();
        assert!(len >= 3);
        let points = self.polygon.points_mut();
        let mut tangled = true;
        let mut swaps = 0;
        while tangled {
            tangled = false;
            let l1 = Line::new(points[len-1], points[0]);
            for j in 1..len-2 {
                let l2 = Line::new(points[j], points[j+1]);
                if l1.intersects(&l2) {
                    points.swap(0, j);
                    swaps += 1;
                    tangled = true;
                    break;
                }
            }
            for i in 0..len-3 {
                for j in i+2..len-1 {
                    let l1 = Line::new(points[i], points[i+1]);
                    let l2 = Line::new(points[j], points[j+1]);
                    if l1.intersects(&l2) {
                        points.swap(i+1, j);
                        swaps += 1;
                        tangled = true;
                        break;
                    }
                }
            }
            if swaps >= 20_000 { // TODO: choose a good limit; this limit should be dependent on the size of the polygon
                //print!("*");
                return None
            }
        }
        //println!("swaps: {}", swaps);
        Some(swaps)
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
    pub fn points(&self) -> &[Point] {
        self.polygon.points()
    }
    fn _edge_tuples<'a>(&self, size: usize) -> SubsetIter {
        assert!(self.polygon.size() >= size);
        SubsetIter {
            size: size,
            max: self.polygon.size() - 1,
            state: SubSetIterState::New
        }
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
        let mut cache = self._vc_dimension_cache.borrow_mut(); // TODO: change to try_borrow_mut; but this function is unstable
        *cache = Some(vc);

        (vc_dim as u8, shattered_subset)
    }
    pub fn vc_dimension(&self) -> u8 {
        let (vc_dim, _) = self._compute_vc_dimension();
        vc_dim
    }
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
    fn export_ipe<W: Write>(&self, mut w: W, scale: f64) -> std::io::Result<()>;
}
impl IpeExport for VcDim {
    fn export_ipe<W: Write>(&self, mut w: W, scale: f64) -> std::io::Result<()> {
        try!(write!(w, r#"<ipe version="70206" creator="libvcdim">
<ipestyle name="vc-poly">
<symbol name="vc-point" transformations="translations">
<path fill="blue">-1.8 -1.8 m 1.8 -1.8 l 1.8 1.8 l -1.8 1.8 l h</path>
</symbol>
<symbol name="vc-point(s)" transformations="translations">
<path fill="sym-stroke"> -1.8 -1.8 m 1.8 -1.8 l 1.8 1.8 l -1.8 1.8 l h</path>
</symbol>
<color name="red" value="1 0 0"/>
<color name="green" value="0 1 0"/>
<color name="blue" value="0 0 1"/>
<color name="yellow" value="1 1 0"/>
<color name="orange" value="1 0.647 0"/>
<color name="purple" value="0.627 0.125 0.941"/>
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
    Malformed
}
impl From<std::io::Error> for IpeImportError {
    fn from(err: std::io::Error) -> Self {
        IpeImportError::IoError(err)
    }
}
pub trait IpeImport {
    fn import_ipe<R: Read>(mut r: R, scale: f64) -> Result<VcDim, IpeImportError>;
}
impl IpeImport for VcDim {
    fn import_ipe<R: Read>(mut r: R, scale: f64) -> Result<VcDim, IpeImportError> {
        // TODO?: should this also import the shattered subset?
        let mut file_contents = String::new();
        try!(r.read_to_string(&mut file_contents));
        // 608
        //println!("{}", &file_contents[608..1000]);
        if let Some(idx_start) = file_contents.find("<path>") {
            let idx_start = idx_start + 6; // 6 == "<path>".len()
            if let Some(idx_end) = file_contents[idx_start..].find("</path>") {
                // all points are listed between indices idx_start and idx_end
                let points_str = &file_contents[idx_start..(idx_start+idx_end)].trim();
                //println!("{}", points_str);
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
                    let vcd = VcDim::new(polygon);
                    return Ok(vcd);
                }
            }
        }
        Err(IpeImportError::Malformed)
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
}
