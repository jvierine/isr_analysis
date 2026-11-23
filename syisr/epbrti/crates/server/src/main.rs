//! Versioned shared HDF5 state, optimistic revisions, atomic snapshot publication.
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tower_http::services::ServeDir;
#[derive(Serialize, Deserialize, Clone)]
struct Meta {
    revision: u64,
    drift_m_s: f64,
    upward_m_s: f64,
    nx: usize,
    ny: usize,
    updated_unix_s: u64,
    #[serde(default = "original_fit")]
    fit_version: String,
}
fn original_fit() -> String {
    "original".into()
}
struct Store {
    meta: Meta,
    values: Vec<f32>,
    root: PathBuf,
}
fn read_h5(path: &Path) -> Result<(Meta, Vec<f32>), Box<dyn std::error::Error>> {
    let h = hdf5::File::open(path)?;
    let raw = h.dataset("metadata_json")?.read_raw::<u8>()?;
    Ok((
        serde_json::from_slice(&raw)?,
        h.dataset("state")?.read_raw::<f32>()?,
    ))
}
fn write_h5(path: &Path, meta: &Meta, v: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let h = hdf5::File::create(path)?;
    h.new_dataset_builder()
        .with_data(&serde_json::to_vec(meta)?)
        .create("metadata_json")?;
    h.new_dataset::<f32>()
        .shape((2, 3, meta.ny, meta.nx))
        .deflate(4)
        .create("state")?
        .write_raw(v)?;
    h.flush()?;
    Ok(())
}
fn binary(s: &Store) -> Response {
    let mut b = Vec::with_capacity(s.values.len() * 4);
    for v in &s.values {
        b.extend(v.to_le_bytes());
    }
    let mut r = b.into_response();
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        "application/octet-stream".parse().unwrap(),
    );
    r.headers_mut()
        .insert("x-revision", s.meta.revision.to_string().parse().unwrap());
    r.headers_mut()
        .insert("x-fit-version", s.meta.fit_version.parse().unwrap());
    r.headers_mut()
        .insert("x-drift-m-s", s.meta.drift_m_s.to_string().parse().unwrap());
    r.headers_mut().insert(
        "x-upward-m-s",
        s.meta.upward_m_s.to_string().parse().unwrap(),
    );
    r.headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    r
}
async fn load(State(s): State<Arc<Mutex<Store>>>) -> Response {
    binary(&s.lock().unwrap())
}
async fn meta(State(s): State<Arc<Mutex<Store>>>) -> Response {
    axum::Json(s.lock().unwrap().meta.clone()).into_response()
}
async fn save(
    State(shared): State<Arc<Mutex<Store>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Same-origin browser writes only; optimistic lock prevents silent multi-tab overwrite.
    if let Some(origin) = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        if origin != "https://juha.no"
            && origin != "http://127.0.0.1:8736"
            && origin != "http://localhost:8736"
        {
            return (StatusCode::FORBIDDEN, "Origin rejected").into_response();
        }
    }
    let mut s = shared.lock().unwrap();
    let rev = headers
        .get("if-match")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    if rev != Some(s.meta.revision) {
        return (
            StatusCode::CONFLICT,
            "Another session saved changes. Download your state before reloading.",
        )
            .into_response();
    }
    let fit = headers
        .get("x-fit-version")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("original");
    if fit != s.meta.fit_version {
        return (StatusCode::CONFLICT,
            "The starting fit has been refined. Download any pending edits, then refresh this page before saving.").into_response();
    }
    if body.len() != s.values.len() * 4 {
        return (StatusCode::BAD_REQUEST, "Wrong state size").into_response();
    }
    let v: Vec<f32> = body
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let n = v.len() / 2;
    if v.iter().any(|v| !v.is_finite())
        || v[..n].iter().any(|v| *v < 0. || *v > 1.)
        || v[n..].iter().any(|v| v.abs() > 10000.)
    {
        return (StatusCode::BAD_REQUEST, "Invalid model values").into_response();
    }
    let speed = headers
        .get("x-drift-m-s")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(s.meta.drift_m_s);
    if !speed.is_finite() || speed < 20. || speed > 400. {
        return (StatusCode::BAD_REQUEST, "Invalid drift").into_response();
    }
    let mut m = s.meta.clone();
    m.revision += 1;
    m.drift_m_s = speed;
    m.updated_unix_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let temp = s
        .root
        .join("snapshots")
        .join(format!("{:08}.tmp.h5", m.revision));
    let dest = s
        .root
        .join("snapshots")
        .join(format!("{:08}.h5", m.revision));
    if let Err(e) = write_h5(&temp, &m, &v).and_then(|_| {
        fs::rename(&temp, &dest)?;
        let tmp = s.root.join("current.tmp");
        fs::write(&tmp, m.revision.to_string())?;
        fs::rename(tmp, s.root.join("current"))?;
        Ok(())
    }) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Save failed: {e}"),
        )
            .into_response();
    }
    s.meta = m.clone();
    s.values = v;
    axum::Json(m).into_response()
}

// Preserve every changed or painted cell exactly. Update only cells that still
// equal the original starting fit, including outside the visible supported area.
fn refine_untouched(current: &mut [f32], old: &[f32], new: &[f32]) -> usize {
    let n = current.len() / 2;
    let mut changed = 0;
    for i in 0..n {
        if current[n + i] == 0. && (current[i] - old[i]).abs() <= 1e-6 {
            current[i] = new[i];
            changed += 1;
        }
    }
    changed
}

fn upgrade_fit(
    root: &Path,
    m: &mut Meta,
    v: &mut Vec<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = root.join("refined_seed.h5");
    if !path.exists() || m.fit_version != "original" {
        return Ok(());
    }
    let (next, refined) = read_h5(&path)?;
    let (_, original) = read_h5(&root.join("seed.h5"))?;
    if refined.len() != v.len()
        || original.len() != v.len()
        || next.nx != m.nx
        || next.ny != m.ny
        || refined.iter().any(|x| !x.is_finite())
        || refined[..v.len() / 2].iter().any(|x| *x < 0. || *x > 1.)
    {
        return Err("Incompatible refined fit".into());
    }
    let count = refine_untouched(v, &original, &refined);
    m.fit_version = next.fit_version;
    m.revision += 1;
    m.updated_unix_s = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let temp = root
        .join("snapshots")
        .join(format!("{:08}.tmp.h5", m.revision));
    let dest = root.join("snapshots").join(format!("{:08}.h5", m.revision));
    if dest.exists() {
        return Err("Refinement revision already exists".into());
    }
    write_h5(&temp, m, v)?;
    fs::rename(temp, dest)?;
    fs::write(root.join("current.tmp"), m.revision.to_string())?;
    fs::rename(root.join("current.tmp"), root.join("current"))?;
    println!(
        "Refined {count} untouched cells; retained {} edited cells in revision {}",
        v.len() / 2 - count,
        m.revision
    );
    Ok(())
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::var("EPBRTI_DATA").unwrap_or("/mnt/shovel/epbrti".into()));
    fs::create_dir_all(root.join("snapshots"))?;
    let source = if root.join("current").exists() {
        let n: u64 = fs::read_to_string(root.join("current"))?.trim().parse()?;
        root.join("snapshots").join(format!("{n:08}.h5"))
    } else {
        root.join("seed.h5")
    };
    let (mut m, mut v) = read_h5(&source)?;
    if v.len() != m.nx * m.ny * 6 {
        return Err("Invalid saved dimensions".into());
    }
    upgrade_fit(&root, &mut m, &mut v)?;
    let shared = Arc::new(Mutex::new(Store {
        meta: m,
        values: v,
        root: root.clone(),
    }));
    let app = Router::new()
        .route("/epbrti/api/state", get(load).post(save))
        .route("/epbrti/api/meta", get(meta))
        .nest_service("/epbrti/data", ServeDir::new(root.join("public")))
        .nest_service(
            "/epbrti",
            ServeDir::new(std::env::var("EPBRTI_WEB").unwrap_or("web".into())),
        )
        .layer(DefaultBodyLimit::max(8 * 1024 * 1024))
        .with_state(shared);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8736").await?;
    println!("EPB RTI ready on 127.0.0.1:8736");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refinement_preserves_painted_and_changed_cells() {
        let old = vec![0.5, 0.5, 0.5, 0., 0., 0.];
        let new = vec![0.2, 0.3, 0.4, 0., 0., 0.];
        let mut state = vec![0.5, 0.7, 0.5, 0., 0., -0.1];
        assert_eq!(refine_untouched(&mut state, &old, &new), 1);
        assert_eq!(state, vec![0.2, 0.7, 0.5, 0., 0., -0.1]);
    }
    #[test]
    fn upgrade_is_versioned_and_idempotent() {
        let root = std::env::temp_dir().join(format!("epbrti-upgrade-{}", std::process::id()));
        fs::create_dir_all(root.join("snapshots")).unwrap();
        let mut m = Meta {
            revision: 7,
            drift_m_s: 120.,
            upward_m_s: 0.,
            nx: 1,
            ny: 1,
            updated_unix_s: 0,
            fit_version: original_fit(),
        };
        let old = vec![0.5, 0.5, 0.5, 0., 0., 0.];
        write_h5(&root.join("seed.h5"), &m, &old).unwrap();
        let mut next = m.clone();
        next.fit_version = "finer-test".into();
        write_h5(
            &root.join("refined_seed.h5"),
            &next,
            &[0.2, 0.3, 0.4, 0., 0., 0.],
        )
        .unwrap();
        let mut state = vec![0.5, 0.7, 0.5, 0., 0., -0.1];
        upgrade_fit(&root, &mut m, &mut state).unwrap();
        assert_eq!(m.revision, 8);
        assert_eq!(m.fit_version, "finer-test");
        let (saved, values) = read_h5(&root.join("snapshots/00000008.h5")).unwrap();
        assert_eq!(saved.revision, 8);
        assert_eq!(values, vec![0.2, 0.7, 0.5, 0., 0., -0.1]);
        assert_eq!(fs::read_to_string(root.join("current")).unwrap(), "8");
        upgrade_fit(&root, &mut m, &mut state).unwrap();
        assert_eq!(m.revision, 8);
        fs::remove_dir_all(root).unwrap();
    }
}
