//! Provides methods for dealing with the Visibility VC-Dimension of a polygon.
//!
//! # Examples
//!
//! The easiest way to get started is to create a new `VcDim` with a random polygon
//! and compute its VC-Dimension:
//!
//! ```rust
//! extern crate polygon;
//! extern crate vcdim;
//!
//! use vcdim::*;
//! use polygon::generate::Mode;
//!
//! fn main() {
//!     // Creates a `VcDim` struct of a polygon with 50 vertices
//!     // generated using the mode `QuickStarLike`.
//!     let v = VcDim::with_random_polygon(50, Mode::QuickStarLike);
//!     // Returns the Visibility VC-Dimension.
//!     let vc_dim = v.vc_dimension();
//!
//!     println!("VC-Dimension: {}", vc_dim);
//!     assert!(vc_dim < 7);
//! }
//! ```

#![warn(missing_docs)]
#![cfg_attr(feature = "bench", feature(test))]

#[cfg(all(feature = "bench", test))]
extern crate test;

extern crate polygon;
extern crate rand;

use polygon::*;
use std::io::{Read, Write};
use std::cell::RefCell;

/// Wraps a `polygon::Polygon`, provides methods for the Visibility VC-Dimension.
#[derive(Debug)]
pub struct VcDim {
    /// The polygon of which the VC dimension is considered.
    ///
    /// The polygon is not pub because mutating it without resetting the cache
    /// and recalculation the visibility matrix leads to wrong results.
    polygon: Polygon,
    /// Visibility matrix of the polygon.
    ///
    /// The bool `visible[i][j]` is true iff `points[i]` sees `points[j]`.
    /// This matrix is symmetrical.
    visible: Vec<Vec<bool>>,
    // Holds an u8 giving the VC-dimension
    // and a Vec<usize> giving the indices of an maximal shattered subset.
    _vc_dimension_cache: RefCell<Option<(u8, Vec<usize>)>>
}
impl VcDim {
    /// Creates a new `VcDim` struct of the given `Polygon`.
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
    /// Creates a new `VcDim` struct with a randomly generated polygon with `n` edges.
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
    /// Generates a new random polygon that has the same number of edges as `self.polygon`.
    pub fn randomize_polygon(&mut self, mode: generate::Mode) {
        self.polygon.randomize(mode);

        self._reset_caches();
        self._calculate_visibility();
    }
    /// Trims the coordinates of all points of the polygon.
    pub fn trim_coordinates(&mut self, dec_places: i8) {
        self.polygon.trim_coordinates(dec_places);

        self._reset_caches();
        self._calculate_visibility();
    }
    /// Resets the VC-Dimenson cache and the visibility matrix.
    ///
    /// Call it every time self.polygon is mutated!
    fn _reset_caches(&mut self) {
        // reset the cache
        self._vc_dimension_cache = RefCell::new(None);

        // The following lines are needed because of
        // an assumption by `_calculate_visibility`.
        for v in &mut self.visible.iter_mut() {
            for e in &mut v.iter_mut() {
                *e = true;
            }
        }
    }
    fn _calculate_visibility(&mut self) {
        // !!! Assumes that self.visible is intialised with 'true' set for each element !!!
        // TODO: this function is needlessly slow!
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
                if !self.polygon.contains(mid) { // this call takes O(n) time
                    self.visible[i][j] = false;
                    self.visible[j][i] = false;
                    continue; // they don't see each other
                }
                for l in self.polygon.edges() { // O(n) iterations
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
     // TODO: should it panic (or whatever) instead?
     //         don't panic (or rewrite the minimize example)!
     //         the best possibility is probably to return an Option or Result.
    pub fn is_shattered(&self, p: &[Point]) -> bool {
        let mut indices = Vec::with_capacity(p.len());
        for i in 0..p.len() {
            if let Some(idx) = self.polygon.points().iter().position(|&x| x == p[i]) {
                indices.push(idx);
            } else {
               // panic!(format!("Point #{} ({:?}) is not a vertex of the polygon!", i, p[i]));
                return false;
            }
        }
        self._is_shattered(&indices)
    }
    /// Computes the region (a polygon) that is visible from the vertex `v`.
    ///
    /// # Panics
    /// Panics if `v` is not a vertex of `self.polygon()`.
    pub fn visibility_region_of(&self, v: Point ) -> Polygon {
        // Using the method described in  chapter 8 of "Art Gallery Theorems and Algorithms".
        // http://cs.smith.edu/~jorourke/books/ArtGalleryTheorems/art.html
        #[derive(Debug, PartialEq, Eq)]
        enum State { Push, Pop, Wait, Stop }
        impl State {
            // In the paper ("Visibility of a Simple Polygon from a Point" by Joe, Simpson)
            // v: z (vision point) This is always equal to points[0]
            // points: v, n (vertices + length)
            // bounding_vertices: s, t (stack of currently visible points + length)
            // idx: i
            // ccw: ccw
            // w: w as a pair of a bool and a point
            fn push(v: Point, points: &[Point], bounding_vertices: &mut Vec<(Point, Angle)>, idx: usize, ccw: &mut bool, w: &mut (bool, Point), angles: &[Angle]) -> (State, usize) {
                let mut i = idx;
                if angles[i+1] <= Angle::full_turn() { // TODO falls i == n-1, so läuft der Index über
                    i += 1;
                    bounding_vertices.push((points[i], angles[i]));
                    if i == points.len()-1 { // TODO hmm... the paper has points[n] == points[0]
                        return (State::Stop, i); // the number is irrelevant
                    } else if angles[i+1] < angles[i] && Angle::is_right_turn(&points[i-1], &points[i], &points[i+1]) {
                        *ccw = true;
                        // add the direction of zv_i to v_i
                        *w = (true, points[i] + points[i] - v);
                        return (State::Wait, i);
                    } else if angles[i+1] < angles[i] && Angle::is_left_turn(&points[i-1], &points[i], &points[i+1]) {
                        return (State::Pop, i);
                    }
                } else {
                    if bounding_vertices.last().unwrap().1 < Angle::full_turn() {
                        // TODO push intersection of v_{i}v_{i+1} and line zv_0
                        let s_t = Line::new(points[i], points[i+1]).line_intersection_with(&Line::new(v, points[1])).expect("should intersect");
                        bounding_vertices.push((s_t, Angle::full_turn()));
                    }

                    *ccw = false;
                    *w = (false, points[1]); // TODO
                    return (State::Wait, i);
                }
                (State::Push, i)
            }
            fn pop(v: Point, points: &[Point], bounding_vertices: &mut Vec<(Point, Angle)>, idx: usize, ccw: &mut bool, w: &mut (bool, Point), angles: &[Angle]) -> (State, usize) {
                let mut i = idx;
                let indices = (0..bounding_vertices.len()).rev().skip(1);
                let mut j = 0; // init as zero because the compiler can't prove that this always gets initialized by the loop
                for k in indices {
                    // scan backwards until one of two conditions is met
                    let angle_sj = bounding_vertices[k].1;
                    let angle_sj1 = bounding_vertices[k+1].1;
                    if angle_sj < angles[i+1] && angles[i+1] <= angle_sj1 ||
                        angles[i+1] <= angle_sj && angle_sj == angle_sj1
                        && Line::new(points[i], points[i+1]).intersects(&Line::new(bounding_vertices[k].0, bounding_vertices[k+1].0))
                        && points[i] != bounding_vertices[k+1].0 { // this last condition is a special case not considered in the paper
                        j = k;
                        break;
                    }
                }
                let j = j;
                if bounding_vertices[j].1 < angles[i+1] {
                    i += 1;
                    bounding_vertices.truncate(j+2); // retain all vertices until index j+1
                    let l1 = Line::new(bounding_vertices[j].0, bounding_vertices[j+1].0);
                    let l2 = Line::new(v, points[i]);
                    let intersection = l1.line_intersection_with(&l2).expect("the lines intersect");
                    bounding_vertices.pop();
                    bounding_vertices.push((intersection, angles[i]));
                    bounding_vertices.push((points[i], angles[i]));

                    if i == points.len()-1 { // TODO hmm... the paper has points[n] == points[0]
                        (State::Stop, i) // the number is irrelevant
                    } else if angles[i+1] >= angles[i] && Angle::is_right_turn(&points[i-1], &points[i], &points[i+1]) {
                        (State::Push, i)
                    } else if angles[i+1] > angles[i] && Angle::is_left_turn(&points[i-1], &points[i], &points[i+1]) {
                        *ccw = false;
                        *w = (false, points[i]);
                        bounding_vertices.pop();
                        (State::Wait, i)
                    } else {
                        (State::Pop, i)
                    }
                } else {
                    if angles[i+1] == bounding_vertices[j].1 && angles[i+2] > angles[i+1] && Angle::is_right_turn(&points[i], &points[i+1], &points[i+2]) {
                        i += 1;
                        bounding_vertices.push((points[i], angles[i]));
                        (State::Push, i)
                    } else {
                        *ccw = true;
                        let l1 = Line::new(points[i], points[i+1]);
                        let l2 = Line::new(bounding_vertices[j].0, bounding_vertices[j+1].0);
                        if l1.slope() == Slope::Vertical && l2.slope() == Slope::Vertical {
                            eprintln!("{:?}\n{:?}", l1, l2);
                            eprintln!("They intersect: {}\n---", l1.intersects(&l2));
                        }
                        bounding_vertices.truncate(j+1);
                        *w = (false, l1.line_intersection_with(&l2).expect("the line segments intersect"));
                        (State::Wait, i)
                    }
                }
            }
            fn wait(_v: Point, points: &[Point], bounding_vertices: &mut Vec<(Point, Angle)>, idx: usize, ccw: &mut bool, w: &mut (bool, Point), angles: &[Angle]) -> (State, usize) {
                let i = idx + 1;
                let tuple_st = bounding_vertices.last().expect("the stack is not empty").clone(); // TODO remove this clone
                        // the clone is cheap (as it's a `Point`) but it only neccessary because the borrow checker isn't smart enough
                let s_t = tuple_st.0;
                let angle_st = tuple_st.1;
                if points.len() == i+1 {
                    //eprintln!("{} {}", w.0, ccw);
                    //eprintln!("{:?}", l1);
                    //eprintln!("{:?}", l2);
                    return (State::Stop, i);
                }
                let l1 = Line::new(points[i], points[i+1]);
                let l2 = Line::new(s_t, w.1);

                if *ccw && angles[i+1] > angle_st && angle_st > angles[i] {
                    if w.0 { // if we need to check intersection with a ray
                        if l1.intersects_ray(&l2) {
                            let intersection = l1.line_intersection_with(&l2).expect("the lines intersect");
                            bounding_vertices.push((intersection, angle_st));
                            return (State::Push, i);
                        }
                    } else {
                        if l1.intersects(&l2) {
                            let intersection = l1.line_intersection_with(&l2).expect("the lines intersect");
                            bounding_vertices.push((intersection, angle_st));
                            return (State::Push, i);
                        }
                    }
                } else if !*ccw && angles[i+1] <= angle_st && angle_st < angles[i] {
                    if w.0 { // if we need to check intersection with a ray
                        if l1.intersects_ray(&l2) {
                            return (State::Pop, i);
                        }
                    } else {
                        if l1.intersects(&l2) {
                            return (State::Pop, i);
                        }
                    }
                }
                (State::Wait, i)
            }
        }

        // First prepare the polygon such that the points
        // are in counter clockwise order and start with `v`.
        let mut polygon = self.polygon().clone();
        if !polygon.is_ccw() {
            polygon.points_mut().reverse();
        }
        assert!(polygon.is_ccw());

        let points = polygon.points_mut();
        let n = points.len();
        let mut idx;
        if let Some(start_idx) = points.iter().position(|x| *x == v) {
            idx = start_idx;
        } else {
           panic!("`v` ({:?}) should be a vertex of the polygon!", v);
        }
        VcDim::slice_rotate(points, idx);
        // Now we can visit the vertices starting at `v` in ccw order
        // by just iterating over `points`.

        // compute angles for each point
        // angles[i] is the accumulated angle of points[i]
        let mut angles = Vec::with_capacity(n);
        angles.push(Angle::zero()); // angle for `v`
        angles.push(Angle::zero()); // angle for the neighbour of `v` in ccw order
        let angles_2 = points[1..].windows(2).map(|e| {
            Angle::from_points(&e[0], &v, &e[1])
        });
        angles.extend(angles_2);
        // Now these are still relative angles (from points[i-1] to points[i]).
        // We need to accumulate them.
        let mut sum = Angle::zero();
        let angles = angles.into_iter().map(|a| {sum = sum + a; sum})
            .collect::<Vec<_>>();

        // The stack that tentativly holds all points that belong
        // to the visibility polygon along with their angles.
        let mut bounding_vertices = Vec::with_capacity(n);
        bounding_vertices.push((points[0], Angle::zero()));
        bounding_vertices.push((points[1], Angle::zero()));

        //let mut p_idx = points[idx];
        idx = 1;
        let mut state = State::Push;
        //let mut angle = angles[idx];
        //assert_eq!(angle, Angle::zero());
        let mut ccw = true;
        // tuple, where the `Point` gives the direction from s_t (bounding_vertices.last())
        // and the bool determines whether this is a ray (`true`) or a line segment (`false`).
        let mut w = (false, Point::origin());

        if angles[2] < angles[1] {
            w = (true, points[1] + (points[1] - points[0]));
            state = State::Wait;
        }


        loop {
            //eprintln!("{:?}\t{:?}", state, points[idx]);
            let s = match state {
                State::Push => State::push(v, points, &mut bounding_vertices, idx, &mut ccw, &mut w, &angles),
                State::Pop => State::pop(v, points, &mut bounding_vertices, idx, &mut ccw, &mut w, &angles),
                State::Wait => State::wait(v, points, &mut bounding_vertices, idx, &mut ccw, &mut w, &angles),
                State::Stop => break,
            };
            state = s.0;
            idx = s.1;
        }
        //eprintln!("---");
        // points[end_idx] is a neighbour of v
        //bounding_vertices.push((points[end_idx], Angle::from_points(...));

        Polygon::from_points(&bounding_vertices.iter().map(|p| p.0).collect::<Vec<_>>())
    }
    fn slice_rotate<T>(slice: &mut [T], mid: usize) {
        // TODO: when out of nightly: just use
        // slice.rotate(mid);
        let len = slice.len();
        slice.reverse();
        let (a, b) = slice.split_at_mut(len - mid);
        a.reverse();
        b.reverse();
    }
    /// Returns a reference to the polygon.
    pub fn polygon(&self) -> &Polygon {
        &self.polygon
    }
    /// Updates the wrapped polygon with the given one.
    pub fn set_polygon(&mut self, p: Polygon) {
        assert!(p.size() >= 3);
        assert!(p.is_simple());
        self.polygon = p;

        self._reset_caches();
        self._calculate_visibility();
    }
    /// Returns the points of `self.polygon()`.
    ///
    /// Convenience method equal to `self.polygon().points()`.
    // TODO: Is this really equal to `self.polygon().points()`?
    pub fn points(&self) -> &[Point] {
        self.polygon.points()
    }
    /// Returns the visibility matrix of the polygon.
    ///
    /// An entry `self.visible()[i][j]` is `true` iff
    /// vertices number i and j see each other.
    pub fn visible(&self) -> &Vec<Vec<bool>> {
        &self.visible
    }
    #[cfg(feature = "naive_dim")]
    fn _edge_tuples(&self, size: usize) -> SubsetIter {
        assert!(self.polygon.size() >= size);
        SubsetIter {
            size: size,
            max: self.polygon.size() - 1,
            state: SubSetIterState::New
        }
    }
    #[cfg(feature = "naive_dim")]
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
    #[cfg(feature = "naive_dim")]
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
    #[cfg(feature = "naive_dim")]
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
    fn _compute_vc_dimension_subset(&self) -> (u8, Vec<usize>) {
        if let Some((dim, ref subset)) = *self._vc_dimension_cache.borrow() {
            return (dim, subset.to_vec()); // clone subset
        }
        if self.polygon.size() <= 3 { return (0, Vec::new()); }
        let max = self.polygon.size() - 1;
        let mut max_shattered = Vec::with_capacity(6);
        let mut pts = Vec::with_capacity(6);
        pts.push(0);

        #[cfg(feature = "print_info")]
        let mut subset_counts = [0, 0, 0, 0, 0, 0]; // counts how many subsets are checked
                                                    // index i counts the subsets of size i+1
        loop {
            #[cfg(feature = "print_info")]
            {subset_counts[pts.len()-1] += 1;}
            let last = *pts.last().expect("pts.len() should always be greater than 0.");
            if self._is_shattered(&pts[..]) {
                if pts.len() > max_shattered.len() {
                    max_shattered.clone_from(&pts);
                }
                if last != max {
                    pts.push(last);
                }
            }
            if last == max {
                pts.pop();
            }
            if pts.is_empty() {
                break;
            }
            *pts.last_mut().expect("We just checked that `pts` is not empty.") += 1;
        }

        #[cfg(feature = "print_info")]
        {
            // print as one csv line
            // n,d,set_s1,set_s1a,set_s2,set_s2a,set_s3,set_s3a,set_s4,set_s4a,set_s5,set_s5a,set_s6,set_s6a,set,set_a,fraction
            let mut sum0 = 0;
            let mut sum1 = 0;
            print!("{},{},", self.polygon.size(), max_shattered.len());
            for i in 1..max_shattered.len()+2 {
                sum0 += subset_counts[i-1];
                sum1 += Self::_binom(i, self.polygon.size());
                print!("{},{},", subset_counts[i-1], Self::_binom(i, self.polygon.size()));
            }
            for _ in max_shattered.len()+2..7 {
                print!("{},{},", 0, 0); // None of both methods considers bigger sets
            }
            println!("{},{},{}", sum0, sum1, sum0 as f64/sum1 as f64);
        }

        let vc = (max_shattered.len() as u8, max_shattered.to_vec()); // copy shattered_subset
        let mut cache = self._vc_dimension_cache.borrow_mut(); // panics if the cache is currently borrowed (should not happen!)
        *cache = Some(vc);

        (max_shattered.len() as u8, max_shattered)
    }
    // For bot <= 7 this works as expected when top <= 500.
    // If the numbers are too heigh, (in release mode) this will silently overflow.
    #[cfg(feature = "print_info")]
    fn _binom(bot: usize, top: usize) -> u64 { // types are quite arbitrary
        let mut prod = top as u64;
        for i in top+1-bot..top {
            prod *= i as u64;
        }
        for i in 2..bot+1 {
            prod /= i as u64;
        }
        prod
    }
    fn _compute_max_shattered_subsets(&self) -> Vec<Vec<usize>> {
        if self.polygon.size() <= 3 { return vec![vec![]]; }
        let max = self.polygon.size() - 1;
        let mut max_shattered_size = 0;
        let mut max_shattered = vec![vec![]];
        let mut pts = Vec::with_capacity(6);
        pts.push(0);
        loop {
            let last = *pts.last().expect("pts.len() should always be greater than 0.");
            if self._is_shattered(&pts[..]) {
                if pts.len() > max_shattered_size {
                    max_shattered_size = pts.len();
                    max_shattered = Vec::new();
                    max_shattered.push(pts.clone());
                } else if pts.len() == max_shattered_size {
                    max_shattered.push(pts.clone());
                }
                if last != max {
                    pts.push(last);
                }
            }
            if last == max {
                pts.pop();
            }
            if pts.is_empty() {
                break;
            }
            *pts.last_mut().expect("We just checked that `pts` is not empty.") += 1;
        }

        if self._vc_dimension_cache.borrow().is_none() {
            // cache the first found shattered subset
            let vc = (max_shattered[0].len() as u8, max_shattered[0].to_vec()); // copy shattered_subset
            let mut cache = self._vc_dimension_cache.borrow_mut(); // panics if the cache is currently borrowed (should not happen!)
            *cache = Some(vc);
        }

        max_shattered
    }
    fn _compute_max_shattered_subsets_limit(&self, limit: usize) -> Vec<Vec<usize>> {
        assert!(limit > 0);
        if self.polygon.size() <= 3 { return vec![vec![]]; }
        let max = self.polygon.size() - 1;
        let mut max_shattered_size = 0;
        let mut max_shattered = vec![vec![]];
        let mut pts = Vec::with_capacity(6);
        pts.push(0);
        loop {
            let last = *pts.last().expect("pts.len() should always be greater than 0.");
            if self._is_shattered(&pts[..]) {
                if pts.len() > max_shattered_size {
                    max_shattered_size = pts.len();
                    max_shattered = Vec::new();
                    max_shattered.push(pts.clone());
                } else if pts.len() == max_shattered_size && max_shattered.len() < limit {
                    max_shattered.push(pts.clone());
                }
                if last != max {
                    pts.push(last);
                }
            }
            if last == max {
                pts.pop();
            }
            if pts.is_empty() {
                break;
            }
            *pts.last_mut().expect("We just checked that `pts` is not empty.") += 1;
        }

        if self._vc_dimension_cache.borrow().is_none() {
            // cache the first found shattered subset
            let vc = (max_shattered[0].len() as u8, max_shattered[0].to_vec()); // copy shattered_subset
            let mut cache = self._vc_dimension_cache.borrow_mut(); // panics if the cache is currently borrowed (should not happen!)
            *cache = Some(vc);
        }

        max_shattered
    }
    /// Returns all maximum shattered subsets of `self.polygon`.
    ///
    /// Caches the first found subset.
    /// If `self.vc_dimension` or `self.max_shattered_subset` are called afterwards,
    /// they just return the cached value.
    pub fn all_max_shattered_subsets(&self) -> Vec<Vec<Point>> {
        let subset_indices = self._compute_max_shattered_subsets();
        subset_indices.into_iter().map(|subset| {
            subset.into_iter().map(|i| { self.points()[i] }).collect()
        }).collect()
    }
    /// Returns `n` many maximum shattered subsets of `self.polygon`.
    ///
    /// This function saves space vs. `self.all_max_shattered_subsets`
    /// as only up to `n` Vector entries must be saved.
    ///
    /// Caches the first found subset.
    /// If `self.vc_dimension` or `self.max_shattered_subset` are called afterwards,
    /// they just return the cached value.
    pub fn first_n_max_shattered_subsets(&self, n: usize) -> Vec<Vec<Point>> {
        let subset_indices = self._compute_max_shattered_subsets_limit(n);
        subset_indices.into_iter().map(|subset| {
            subset.into_iter().map(|i| { self.points()[i] }).collect()
        }).collect()
    }
    /// Returns the VC-dimension of `self.polygon`.
    ///
    /// Caches the result of the computation.
    /// If `self.vc_dimension` or `self.max_shattered_subset` was called before,
    /// just returns the cached value.
    pub fn vc_dimension(&self) -> u8 {
        #[cfg(not(feature = "naive_dim"))]
        let (vc_dim, _) = self._compute_vc_dimension_subset();
        #[cfg(feature = "naive_dim")]
        let (vc_dim, _) = self._compute_vc_dimension_naive();
        vc_dim
    }
    /// Returns a maximum shattered subset of `self.polygon`.
    ///
    /// Caches the result of the computation.
    /// If `self.vc_dimension` or `self.max_shattered_subset` was called before,
    /// just returns the cached value.
    pub fn max_shattered_subset(&self) -> Vec<Point> {
        #[cfg(not(feature = "naive_dim"))]
        let (_, subset) = self._compute_vc_dimension_subset();
        #[cfg(feature = "naive_dim")]
        let (_, subset) = self._compute_vc_dimension_naive();
        subset.into_iter().map(|i| self.polygon.points()[i]).collect()
    }
    /// Tests if this polygon is among the smallest (i.e. low number of vertices)
    /// known polygons with its VC-Dimension.
    ///
    /// For VC-Dimension `d < 6` this returns if this is a minimum polygon.
    pub fn is_small(&self) -> bool {
        let n = self.polygon.size();
        match self.vc_dimension() {
            0 => n == 3,
            1 => n == 4,
            2 => n == 6,
            3 => n == 8,
            4 => n == 16,
            5 => n == 32,
            6 => true,
            _ => true,
        }
    }
}
#[cfg(feature = "naive_dim")]
enum SubSetIterState {
    New,
    Subset(Vec<usize>),
    Stopped
}
#[cfg(feature = "naive_dim")]
struct SubsetIter {
    size: usize,
    max: usize,
    state: SubSetIterState
}
#[cfg(feature = "naive_dim")]
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

/// Export something as an [ipe](http://ipe.otfried.org/) file.
pub trait IpeExport {
    /// The Error type that is returned in case the export fails.
    type Error;

    /// Export an ipe file from the given `Write`er `w`.
    ///
    /// The parameter `scale` allows to multiply each input point with this value.
    fn export_ipe<W: Write>(&self, w: W, scale: f64) -> std::io::Result<Self::Error>;
}
impl IpeExport for VcDim {
    type Error = ();

    fn export_ipe<W: Write>(&self, mut w: W, scale: f64) -> std::io::Result<()> {
        write!(w, r#"<ipe version="70206" creator="libvcdim">
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
<path>"#)?;
        let points = self.polygon.points();
        if points.len() > 0 {
            write!(w, "{} {} m ", points[0].x * scale, points[0].y * scale)?;
        }
        for p in points.iter().skip(1) {
            write!(w, "{} {} l ", p.x * scale, p.y * scale)?;
        }
        write!(w, "h</path>\n")?;
        for p in self.max_shattered_subset() {
            write!(w, r#"<use name="vc-point" pos="{} {}"/>"#, p.x * scale, p.y * scale)?;
        }
        for (i, p) in points.iter().enumerate() {
            if self.max_shattered_subset().contains(p) {
                write!(w, r#"<text pos="{} {}" size="6" stroke="blue" matrix="1 0 0 1 2 0" valign="center">{}</text>"#, p.x * scale, p.y * scale, i)?;
            } else {
                write!(w, r#"<text pos="{} {}" size="3" valign="center">{}</text>"#, p.x * scale, p.y * scale, i)?;
            }
        }
        write!(w, "</page>\n</ipe>")
    }
}
/// Error from importing an ipe file to `VcDim`.
#[derive(Debug)]
pub enum IpeImportError { //TODO?: doesn't impl Error for now
    /// An IoError occured.
    IoError(std::io::Error),
    /// The imported file is not valid.
    Malformed,
    /// A shattered subset was specified in the file
    /// that is in fact not shattered.
    SubsetNotShattered(VcDim)
}
impl std::fmt::Display for IpeImportError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            &IpeImportError::IoError(ref e) => write!(fmt, "An IoError occurred: {}", e),
            &IpeImportError::Malformed => write!(fmt, "The file  is malformed."),
            &IpeImportError::SubsetNotShattered(_) => write!(fmt, "The provided subset is not shattered."),
        }
    }
}
impl From<std::io::Error> for IpeImportError {
    fn from(err: std::io::Error) -> Self {
        IpeImportError::IoError(err)
    }
}
impl std::error::Error for IpeImportError {
    fn description(&self) -> &str {
        match self {
            &IpeImportError::IoError(_) => "IoError",
            &IpeImportError::Malformed => "File malformed",
            &IpeImportError::SubsetNotShattered(_) => "Subset not shattered",
        }
    }
}
/// Import something from an [ipe](http://ipe.otfried.org/) file.
pub trait IpeImport {
    /// The Error type that is returned in case the import fails.
    type Error;

    /// Import an ipe file from the given `Read`er `r`.
    ///
    /// The parameter `scale` allows to multiply each input point with this value.
    fn import_ipe<R: Read>(r: R, scale: f64) -> Result<Self, Self::Error>
        where Self: std::marker::Sized;
}
impl IpeImport for VcDim {
    type Error = IpeImportError;

    fn import_ipe<R: Read>(mut r: R, scale: f64) -> Result<VcDim, IpeImportError> {
        // TODO: do parsing with regex.
        let mut file_contents = String::new();
        r.read_to_string(&mut file_contents)?;
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
                            let x = split[0].parse::<f64>().expect("Couldn't parse x-coord as f64") * scale;
                            let y = split[1].parse::<f64>().expect("Couldn't parse y-coord as f64") * scale;
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
                let x = coords[0].parse::<f64>().expect("Couldn't parse x-coord as f64") * scale;
                let y = coords[1].parse::<f64>().expect("Couldn't parse y-coord as f64") * scale;
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
    #[cfg(feature = "naive_dim")]
    use polygon::generate::Mode;

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
    #[cfg(feature = "naive_dim")]
    fn _edge_tuples_from_previous_set_works() {
        for size in 10..20 {
            let v = VcDim::with_random_polygon(size, Mode::QuickStarLike); // TODO: deterministic polygon to save time (only size of polygon is needed)
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
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_fns_are_equal() {
        for size in 3..20 {
            for _ in 0..10 {
                let v = VcDim::with_random_polygon(size, Mode::QuickStarLike);
                let dim1 = v._compute_vc_dimension_naive();
                let dim2 = v._compute_vc_dimension();
                assert_eq!(dim1, dim2);
                let dim3 = v._compute_vc_dimension_subset();
                assert_eq!(dim1, dim3);
            }
        }
    }
}
#[cfg(all(feature = "bench", test))]
mod bench {
    use super::*;
    use test::Bencher;
    use polygon::generate::Mode;

    const MODE: Mode = Mode::QuickStarLike;

    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_naive_10(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(10, MODE);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_10(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(10, MODE);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_subset_10(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(10, MODE);
        b.iter(|| v._compute_vc_dimension_subset());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_naive_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20, MODE);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20, MODE);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_subset_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20, MODE);
        b.iter(|| v._compute_vc_dimension_subset());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_naive_30(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(30, MODE);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_30(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(30, MODE);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_subset_30(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(30, MODE);
        b.iter(|| v._compute_vc_dimension_subset());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_naive_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40, MODE);
        b.iter(|| v._compute_vc_dimension_naive());
    }
    #[bench]
    #[cfg(feature = "naive_dim")]
    fn _compute_vc_dimension_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40, MODE);
        b.iter(|| v._compute_vc_dimension());
    }
    #[bench]
    fn _compute_vc_dimension_subset_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40, MODE);
        b.iter(|| v._compute_vc_dimension_subset());
    }
    #[bench]
    fn _is_shattered_2_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20, MODE);
        b.iter(|| v._is_shattered(&[1, 4]));
    }
    #[bench]
    fn _is_shattered_2_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40, MODE);
        b.iter(|| v._is_shattered(&[1, 4]));
    }
    #[bench]
    fn _is_shattered_2_80(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(80, MODE);
        b.iter(|| v._is_shattered(&[1, 4]));
    }
    #[bench]
    fn _is_shattered_3_20(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(20, MODE);
        b.iter(|| v._is_shattered(&[1, 4, 7]));
    }
    #[bench]
    fn _is_shattered_3_40(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(40, MODE);
        b.iter(|| v._is_shattered(&[1, 4, 7]));
    }
    #[bench]
    fn _is_shattered_3_80(b: &mut Bencher) {
        let v = VcDim::with_random_polygon(80, MODE);
        b.iter(|| v._is_shattered(&[1, 4, 7]));
    }
}
