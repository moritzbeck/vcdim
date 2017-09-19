extern crate quick_xml;
extern crate polygon;
extern crate vcdim;

use quick_xml::writer::Writer;
use quick_xml::events::{Event, BytesStart, BytesEnd, BytesText};
use quick_xml::reader::Reader;
use vcdim::*;
use std::fs::*;
use std::io::Write;
use std::io::BufReader;

fn export_w_visibility_regions<W: Write>(in_file: &str, w: W) {
    let vcd = VcDim::import_ipe(std::fs::File::open(in_file).expect("ipe file not found"), 1.).expect("File is malformed!");
    let mut reader = Reader::from_reader(BufReader::new(std::fs::File::open(in_file).expect("ipe file not found")));
    reader.trim_text(true);
    let mut writer = Writer::new(w);
    let mut buf = Vec::new();

    loop {
        match reader.read_event(&mut buf) {
            Err(e) => panic!("Error at position {}: {:?}", reader.buffer_position(), e),
            Ok(Event::Eof) => break,
            Ok(Event::Start(ref e)) if e.name() == b"ipestyle" => {
                assert!(writer.write_event(Event::Start(BytesStart::borrowed(b"ipestyle", "ipestyle".len()))).is_ok());

                let mut opc = BytesStart::owned(b"opacity".to_vec(), "opacity".len());
                opc.push_attribute(("name", "transparent"));
                opc.push_attribute(("value", "0.2"));
                assert!(writer.write_event(Event::Empty(opc)).is_ok());
            },
            Ok(Event::Start(ref e)) if e.name() == b"page" => {
                assert!(writer.write_event(Event::Start(BytesStart::borrowed(b"page", "page".len()))).is_ok());
                // add "alpha"-layer
                let mut layer = BytesStart::owned(b"layer".to_vec(), "layer".len());
                layer.push_attribute(("name", "alpha"));
                assert!(writer.write_event(Event::Empty(layer)).is_ok());

                // add layers for visibility regions
                for i in 0..vcd.points().len() {
                    let mut layer = BytesStart::owned(b"layer".to_vec(), "layer".len());
                    layer.push_attribute(("name", format!("vis_region_{:02}", i).as_str()));
                    assert!(writer.write_event(Event::Empty(layer)).is_ok());
                }
                // add view with just alpha-layer visible
                let mut view = BytesStart::owned(b"view".to_vec(), "view".len());
                view.push_attribute(("layers", "alpha"));
                view.push_attribute(("active", "alpha"));
                assert!(writer.write_event(Event::Empty(view)).is_ok());
            },
            Ok(Event::End(ref e)) if e.name() == b"page" => {
                // add visibility regions
                for pt in vcd.points().iter().enumerate() {
                    let (i, p) = pt;
                    let vis_region = vcd.visibility_region_of(*p);

                    let mut path_start = BytesStart::owned(b"path".to_vec(), "path".len());
                    path_start.push_attribute(("layer", format!("vis_region_{:02}", i).as_str()));
                    path_start.push_attribute(("stroke", "black"));
                    path_start.push_attribute(("fill", "green"));
                    path_start.push_attribute(("opacity", "transparent"));
                    let path_end = BytesEnd::owned(b"path".to_vec());
                    let mut text = Vec::new();
                    let vertices = vis_region.points();
                    text.append(&mut format!("{} {} m ", vertices[0].x, vertices[0].y).into_bytes());
                    for v in &vertices[1..] {
                        text.append(&mut format!("{} {} l ", v.x, v.y).into_bytes());
                    }
                    text.push(b'h');
                    let text_node = BytesText::owned(text);

                    assert!(writer.write_event(Event::Start(path_start)).is_ok());
                    assert!(writer.write_event(Event::Text(text_node)).is_ok());
                    assert!(writer.write_event(Event::End(path_end)).is_ok());
                }
                assert!(writer.write_event(Event::End(BytesEnd::borrowed(b"page"))).is_ok());
            },
            Ok(e) => assert!(writer.write_event(e).is_ok()),
        }
        buf.clear();
    }
}


fn main() {
    let mut args = std::env::args().skip(1);
    let in_file = if let Some(arg) = args.next() {
        arg
    } else {
        println!("Please provide a file name!");
        return;
    };
    let out_dir = "out";

    // TODO: let `export_w_visibility_regions` return a `Result`.
    export_w_visibility_regions(&in_file, File::create(format!("{}/{}.regions.ipe", out_dir, in_file)).unwrap());
}
