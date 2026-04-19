# app.py  -- MODE 3 (LLM IndoBERT + MiniLM + Rule-based + SNA + Anomali)

import streamlit as st
import pandas as pd
import numpy as np
import io
import csv
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from collections import Counter


# ============================================================
# OPTIONAL: LLM / HF model imports (aman kalau tidak terpasang)
# ============================================================
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoConfig,
    )

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    torch = None
    AutoTokenizer = AutoModelForSequenceClassification = AutoConfig = None

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
# =========================
# HEADER: LOGO + JUDUL
# =========================
LOGO_PATH = "bin.png"   # atau "bin.png"

st.markdown(
    "<h1 style='text-align:center; margin-bottom:8px;'>Sistem Analisis Konten Radikal</h1>",
    unsafe_allow_html=True
)

st.image(LOGO_PATH, width=85, caption=None, use_container_width=False)
st.markdown("<div style='text-align:center; margin-top:-8px;'></div>", unsafe_allow_html=True)

st.markdown("---")



# ============================================================
# KONFIGURASI MODEL HUGGINGFACE
# ============================================================

# 1) Model IndoBERT SENTIMEN (bukan radikal khusus, tapi untuk emosi/negativitas)
INDOBERT_MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"
# id2label di HF: {0: "negative", 1: "neutral", 2: "positive"}

# 2) MiniLM multilingual embeddings (cocok untuk bahasa Indonesia)
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# LEXICON / HEURISTIK MODE KETAT
# ============================================================

lexicon_radikal = [
    # A. Ideologis
    "thaghut",
    "taghut",
    "t4ghut",
    "th4ghut",
    "toghut",
    "tahgut",
    "kafir",
    "k@fir",
    "kaf1r",
    "murtad",
    "munafik",
    "munafiq",
    "zalim",
    "dzalim",
    "dhalim",
    "dholim",
    "jahiliyah",
    "jahiliah",
    "salibis",
    "crusader",
    "darul harb",
    "darul kufr",
    "wilayah perang",
    # B. Kekerasan
    "jihad",
    "j1had",
    "j!had",
    "jihadist",
    "amaliyah",
    "amaliah",
    "amaliyat",
    "istisyhad",
    "istisyhadi",
    "istisyhadiyah",
    "bom syahid",
    "bom bunuh diri",
    "ghazwah",
    "ghazwat",
    "ghozoah",
    "qital",
    "perang badar",
    "serang",
    "hancurkan",
    "habisi",
    "lone wolf",
    "serigala tunggal",
    # C. Struktur ISIS/JAD
    "daulah",
    "dawlah",
    "d4ulah",
    "khilafah",
    "khil4fah",
    "khilafiyah",
    "imarah",
    "amir",
    "ameer",
    "khalifah",
    "waliyul amri",
    "anshar",
    "ansar",
    "anshor",
    "ansharut",
    "muhajir",
    "muhajirin",
    # D. Rekrutmen
    "hijrah",
    "hijroh",
    "h1jrah",
    "baiat",
    "bayat",
    "b@iat",
    "b4iat",
    "istiqamah perjuangan",
    "istiqomah perjuangan",
    "ukhti fillah",
    "akh fi sabilillah",
    "tarhib",
    "targhib",
    "bergabung di jalan ini",
    "tinggalkan negeri kafir",
    # E. Propaganda
    "nashir",
    "nasyir",
    "nasher",
    "bayan",
    "qashidah",
    "anasyid jihad",
    "ghuraba",
    "ghuroba",
    "tauhid murni",
    "tawheed",
    # F. Cluster Simpatisan
    "dhulm",
    "dzulm",
    "#zalim",
    "#khilafah",
    "#dawlah",
    "millah ibrahim",
    "milah ibrahim",
    "millahibrahim",
    "taghutoff",
    "#hancurkanthaghut",
    "syuhada",
    "syahid",
    "syahidah",
    "#lonewolf",
    "aktor tunggal",
    # Tambahan lokal
    "idad",
    "syariat kaffah",
    "amar ma'ruf nahi munkar",
    "amar makruf nahi munkar",
]

kata_kunci_negatif = [
    "serang",
    "hancurkan",
    "musuh",
    "radikal",
    "bom",
    "senjata",
    "perang",
    "habisi",
]
kata_kunci_positif = [
    "damai",
    "toleransi",
    "bersatu",
    "harmoni",
    "cinta",
    "kebaikan",
    "keadilan",
    "kemanusiaan",
]

kata_kerja_ajakan_radikal = [
    "hancurkan",
    "hancurkah",
    "hancur kan",
    "serang",
    "serbu",
    "habisi",
    "tumpas",
    "perangi",
    "perangilah",
    "angkat senjata",
    "siapkan senjata",
    "lawan dengan senjata",
    "bom",
    "ledakkan",
    "tusuk",
    "tembak",
    "tegakkan khilafah",
    "tegakkan daulah",
    "tegakkan syariat kaffah",
    "baiat",
    "berbaiat",
    "berbai'ah",
    "mari baiat",
    "bergabung dengan",
    "bergabung di jalan ini",
    "ikut jihad",
    "berjihad",
    "ayo jihad",
    "wajib jihad",
    "dukung khilafah",
    "dukung daulah",
    "setia kepada khilafah",
    "syahid di medan jihad",
    "siap mati syahid",
]

frasa_penguatan_radikal = [
    "kewajiban setiap muslim sejati",
    "kewajiban kita",
    "jalan kebenaran",
    "satu-satunya jalan",
    "tidak ada pilihan lain selain",
    "ini perintah allah",
    "ini perintah tuhan",
    "demi agama",
    "demi iman",
    "jihad adalah solusi",
    "khilafah adalah solusi",
    "syariat kaffah satu-satunya jalan",
]

konteks_anti_radikal = [
    "bahaya radikalisme",
    "bahaya propaganda",
    "bahaya paham radikal",
    "bahaya ekstremisme",
    "bahaya terorisme",
    "bahaya lone wolf",
    "melawan radikalisme",
    "melawan ekstremisme",
    "melawan terorisme",
    "menolak radikalisme",
    "tolak radikalisme",
    "tolak kekerasan",
    "mencegah radikalisme",
    "pencegahan radikalisme",
    "jangan terpengaruh",
    "jangan terprovokasi",
    "perlu kewaspadaan terhadap",
    "kewaspadaan terhadap",
    "waspada terhadap",
    "antisipasi penyebaran",
    "sebagian orang menyalahgunakan istilah jihad",
    "pendistorsian makna jihad",
    "penyimpangan makna jihad",
    "bukan ajaran islam",
    "tidak sesuai ajaran islam",
    "kontra radikalisme",
    "narasi damai",
    "kontra narasi",
    "edukasi tentang bahaya",
    "sosialisasi bahaya",
    "menjelaskan bahaya",
    "mengkritik paham radikal",
]

# ============================================================
# STOPWORDS
# ============================================================
stopwords_id_manual = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "dalam",
    "pada",
    "dengan",
    "karena",
    "sehingga",
    "agar",
    "untuk",
    "adalah",
    "ialah",
    "bahwa",
    "ini",
    "itu",
    "itu",
    "sudah",
    "telah",
    "akan",
    "atau",
    "juga",
    "saja",
    "lagi",
    "sebagai",
    "serta",
    "kami",
    "kita",
    "saya",
    "aku",
    "anda",
    "kamu",
    "engkau",
    "dia",
    "ia",
    "mereka",
    "para",
    "pun",
    "lah",
    "punya",
    "the",
    "is",
    "are",
    "of",
    "on",
    "in",
    "at",
    "to",
    "for",
    "an",
    "a",
    "and",
    "or",
}

wordcloud_default_sw = set(STOPWORDS)
STOPWORDS_WORDCLOUD = wordcloud_default_sw.union(stopwords_id_manual)
STOPWORDS_TFIDF = list(stopwords_id_manual)

# ============================================================
# LOADER MODEL HUGGINGFACE (CACHED)
# ============================================================


@st.cache_resource(show_spinner=False)
def load_indobert_model():
    """Load IndoBERT sentiment model. Kalau gagal, return (None, None, None)."""
    if not HAS_TRANSFORMERS or torch is None:
        return None, None, None
    try:
        config = AutoConfig.from_pretrained(INDOBERT_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            INDOBERT_MODEL_NAME
        )
        model.eval()
        id2label = (
            config.id2label
            if hasattr(config, "id2label")
            else {i: str(i) for i in range(model.num_labels)}
        )
        return tokenizer, model, id2label
    except Exception as e:
        st.sidebar.warning(
            f"Gagal load IndoBERT model '{INDOBERT_MODEL_NAME}': {e}\n"
            "Model LLM akan dimatikan (fallback ke rule-based saja)."
        )
        return None, None, None


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load MiniLM sentence-transformers untuk SNA / embedding."""
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model
    except Exception as e:
        st.sidebar.warning(
            f"Gagal load embedding model '{EMBEDDING_MODEL_NAME}': {e}\n"
            "SNA akan fallback ke TF-IDF."
        )
        return None


# ============================================================
# UTILITAS: LOAD CSV
# ============================================================


def load_csv_safe(uploaded_file_obj):
    raw_bytes = uploaded_file_obj.read()
    if not raw_bytes:
        st.error("File kosong (0 bytes).")
        return None

    raw_text = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            raw_text = raw_bytes.decode(enc)
            break
        except Exception:
            continue
    if raw_text is None:
        st.error("Gagal decode file.")
        return None

    try:
        dialect = csv.Sniffer().sniff(
            raw_text[:16384], delimiters=[",", ";", "\t", "|"]
        )
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ","

    buf = io.StringIO(raw_text)
    try:
        df = pd.read_csv(buf, delimiter=delimiter)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return None

    df.columns = df.columns.astype(str).str.strip().str.lower()

    col_map = {
        "fb name": "nama",
        "fb_name": "nama",
        "uid": "user_id",
        "follower": "follower",
        "post id": "post_id",
        "post_id": "post_id",
        "konten": "isi_postingan",
        "konten_post": "isi_postingan",
        "post link": "post_link",
        "jmh like": "jumlah_like",
        "jmh comment": "jumlah_komentar",
        "jmh share": "jumlah_share",
        "like": "jumlah_like",
        "likes": "jumlah_like",
        "komentar": "jumlah_komentar",
        "comments": "jumlah_komentar",
        "share": "jumlah_share",
        "shares": "jumlah_share",
        "id": "post_id",
        "user": "user_id",
        "name": "nama",
    }
    df.rename(
        columns={k: v for k, v in col_map.items() if k in df.columns},
        inplace=True,
    )

    required = [
        "user_id",
        "nama",
        "isi_postingan",
        "jumlah_like",
        "jumlah_komentar",
        "jumlah_share",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Kolom wajib hilang dari dataset Anda: {missing}")
        return None

    df["isi_postingan"] = df["isi_postingan"].astype(str).fillna("")
    for col in ["jumlah_like", "jumlah_komentar", "jumlah_share"]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        )
    df["user_id"] = df["user_id"].astype(str)
    df["nama"] = df["nama"].astype(str)

    return df


# ============================================================
# PREPROCESSING TEKS
# ============================================================


def preprocess_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^0-9a-zA-Z\s#]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_radikal_strict(text: str) -> int:
    """
    Rule-based KETAT:
    1. Ada istilah radikal (lexicon_radikal)
    2. DAN ada kata kerja ajakan / glorifikasi
    3. TIDAK ada konteks anti-radikalisme / edukatif.
    """
    t = preprocess_text(text)

    # 1) ada istilah radikal
    has_lex = any(kw in t for kw in lexicon_radikal)
    if not has_lex:
        return 0

    # 2) kalau ada frasa anti-radikalisme → bukan radikal
    if any(fr in t for fr in konteks_anti_radikal):
        return 0

    # 3) cek ajakan / glorifikasi
    has_verb = any(v in t for v in kata_kerja_ajakan_radikal)
    has_praise = any(fr in t for fr in frasa_penguatan_radikal)

    if has_verb or has_praise:
        return 1

    # fallback: banyak istilah radikal, tanpa kata "bahaya/tolak/melawan"
    raw_count = sum(1 for kw in lexicon_radikal if kw in t)
    if raw_count >= 3 and not any(
        k in t for k in ["bahaya", "waspada", "tolak", "melawan", "mencegah"]
    ):
        return 1

    return 0


def detect_sentiment_rule(text: str) -> str:
    t = text.lower()
    if any(k in t for k in kata_kunci_negatif):
        return "NEGATIVE"
    if any(k in t for k in kata_kunci_positif):
        return "POSITIVE"
    return "NEUTRAL"


def plot_wordcloud_from_series(series: pd.Series, title: str = None):
    txt = " ".join(series.dropna().astype(str).tolist())
    if not txt.strip():
        st.info("Tidak ada teks untuk WordCloud.")
        return

    wc = WordCloud(
        width=1000,
        height=400,
        background_color="white",
        stopwords=STOPWORDS_WORDCLOUD,
    ).generate(txt)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    st.pyplot(fig)


# ============================================================
# SNA: GRAPH BUILDER (matrix bisa TF-IDF ataupun Embedding)
# ============================================================


def build_similarity_graph(feature_matrix, df_users, threshold: float = 0.35):
    """
    feature_matrix: ndarray (n_post x dim) dari MiniLM / TF-IDF
    df_users: kolom user_id, nama (index sejajar dengan feature_matrix)
    """
    df_users = df_users.reset_index(drop=True)
    sim = cosine_similarity(feature_matrix)
    G = nx.Graph()

    unique_users = (
        df_users[["user_id", "nama"]].drop_duplicates().set_index("user_id")
    )
    for uid, row in unique_users.iterrows():
        G.add_node(uid, label=row["nama"], posts=0)

    counts = df_users.groupby("user_id").size().to_dict()
    for uid, c in counts.items():
        if uid in G:
            G.nodes[uid]["posts"] = int(c)

    posts_by_user = (
        df_users.groupby("user_id").apply(lambda x: x.index.tolist()).to_dict()
    )

    user_ids = list(posts_by_user.keys())
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            u = user_ids[i]
            v = user_ids[j]
            idx_u = posts_by_user[u]
            idx_v = posts_by_user[v]
            if len(idx_u) == 0 or len(idx_v) == 0:
                continue
            pair_sims = sim[np.ix_(idx_u, idx_v)]
            max_sim = float(pair_sims.max())
            if max_sim >= threshold:
                G.add_edge(u, v, weight=max_sim)
    return G

def cluster_accounts_by_narrative(feature_matrix, df_users, n_clusters=3):
    """
    Mengelompokkan AKUN berdasarkan kemiripan narasi posting radikal.
    - feature_matrix: embedding posting radikal (n_post x dim)
    - df_users: dataframe kolom user_id, nama (sejajar dengan feature_matrix)
    Return: dict {user_id: cluster_id}
    """

    # 1. Rata-rata embedding per akun
    user_embeddings = {}
    for uid in df_users["user_id"].unique():
        idx = df_users[df_users["user_id"] == uid].index
        user_embeddings[uid] = feature_matrix[idx].mean(axis=0)

    user_ids = list(user_embeddings.keys())
    X_user = np.vstack([user_embeddings[u] for u in user_ids])

    # 2. Clustering narasi (cosine)
    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(user_ids)),
        metric="cosine",
        linkage="average"
    )

    labels = clustering.fit_predict(X_user)

    return dict(zip(user_ids, labels))


# ============================================================
# STATUS MODEL DI SIDEBAR
# ============================================================

st.sidebar.subheader("Status Model")

tokenizer_indobert, model_indobert, id2label_indobert = load_indobert_model()
if tokenizer_indobert is not None:
    st.sidebar.success(
        f"IndoBERT SENTIMEN aktif: {INDOBERT_MODEL_NAME.split('/')[-1]}"
    )
else:
    st.sidebar.error(
        "IndoBERT classifier TIDAK aktif. Kolom LLM akan kosong / None."
    )

embedding_model = load_embedding_model()
if embedding_model is not None:
    st.sidebar.success(
        f"MiniLM embeddings aktif: {EMBEDDING_MODEL_NAME.split('/')[-1]}"
    )
else:
    st.sidebar.error(
        "sentence-transformers TIDAK terpasang / embedding gagal.\n"
        "SNA akan memakai TF-IDF."
    )

# ============================================================
# INPUT DATASET
# ============================================================

uploaded = st.sidebar.file_uploader(
    "Upload dataset CSV (format: Fb Name; UID; Follower; Post ID; Konten; Jmh Like; Jmh Comment; Jmh Share) atau variasinya",
    type=["csv"],
)

sim_threshold = st.sidebar.slider(
    "SNA: ambang cosine similarity",
    min_value=0.10,
    max_value=0.80,
    value=0.35,
    step=0.01,
)
node_size_base = st.sidebar.slider(
    "SNA: ukuran dasar node", min_value=10, max_value=60, value=20, step=1
)

if not uploaded:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")
    st.stop()

df = load_csv_safe(uploaded)
if df is None:
    st.stop()

# ============================================================
# PREPROCESSING & FITUR DASAR
# ============================================================

df["clean_text"] = df["isi_postingan"].apply(preprocess_text)
df["total_engagement"] = (
    df["jumlah_like"] + df["jumlah_komentar"] + df["jumlah_share"]
)

# Rule-based KETAT
df["indikator_radikal_rule"] = df["isi_postingan"].apply(is_radikal_strict)
df["sentimen_rule"] = df["isi_postingan"].apply(detect_sentiment_rule)

# TF-IDF (untuk fallback & analitik)
tfidf = TfidfVectorizer(
    max_features=1500, ngram_range=(1, 2), stop_words=STOPWORDS_TFIDF
)
X_tfidf = tfidf.fit_transform(df["clean_text"])

# ============================================================
# LLM: PREDIKSI INDO BERT (SENTIMEN)
# ============================================================


def indo_sentiment_predict(texts):
    """Return label dan skor max-prob untuk list teks."""
    if tokenizer_indobert is None or model_indobert is None:
        return [None] * len(texts), [None] * len(texts)

    all_labels = []
    all_scores = []
    device = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
    model_indobert.to(device)

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer_indobert(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model_indobert(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for p in probs:
            idx = int(np.argmax(p))
            label = id2label_indobert.get(str(idx), id2label_indobert.get(idx, str(idx)))
            all_labels.append(label)
            all_scores.append(float(p[idx]))

    return all_labels, all_scores


df["llm_sentiment_label"], df["llm_sentiment_score"] = indo_sentiment_predict(
    df["clean_text"].tolist()
)

# Konversi ke skor risiko kasar (negatif = tinggi, neutral sedang, positif rendah)
sent2risk = {"negative": 1.0, "neutral": 0.4, "positive": 0.0}
df["llm_risk_score"] = df["llm_sentiment_label"].map(sent2risk).fillna(
    0.0
)

# Kombinasi sederhana rule + LLM untuk skor final (0–1)
# (rule_ketat lebih dominan)
df["skor_radikal_final"] = (
    0.7 * df["indikator_radikal_rule"] + 0.3 * df["llm_risk_score"]
)
df["indikator_radikal_final"] = (df["skor_radikal_final"] >= 0.7).astype(int)

# ============================================================
# Anomaly Detection (engagement)
# ============================================================
iso = IsolationForest(contamination=0.12, random_state=42)
try:
    df["anomali_flag"] = iso.fit_predict(df[["total_engagement"]])
    df["anomali_flag"] = df["anomali_flag"].map(
        {1: "Normal", -1: "Anomali"}
    )
except Exception as e:
    st.warning(f"Anomaly detection gagal: {e}")
    df["anomali_flag"] = "Unknown"

# ============================================================
# Embeddings untuk SNA (MiniLM → fallback TF-IDF)
# ============================================================
if embedding_model is not None:
    try:
        emb_all = embedding_model.encode(
            df["clean_text"].tolist(),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        feature_matrix_sna = emb_all
        sna_source = "MiniLM embeddings"
    except Exception as e:
        st.warning(
            f"Gagal membuat embeddings MiniLM, fallback ke TF-IDF: {e}"
        )
        feature_matrix_sna = X_tfidf.toarray()
        sna_source = "TF-IDF (fallback)"
else:
    feature_matrix_sna = X_tfidf.toarray()
    sna_source = "TF-IDF (fallback)"

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "NLP & WordCloud",
        "Anomaly",
        "LLM Classifier (IndoBERT)",
        "SNA (Radikal saja)",
        "Rekomendasi Cegah Dini",
    ]
)

# ------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.header("Overview Dataset")
    st.markdown(f"- Baris dataset: **{len(df)}**")
    st.markdown(f"- User unik: **{df['user_id'].nunique()}**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preview Data")
        st.dataframe(df.head(50))
    with col2:
        st.subheader("Statistik Singkat (Mode Ketat + LLM)")
        n_rad_rule = int(df["indikator_radikal_rule"].sum())
        n_rad_final = int(df["indikator_radikal_final"].sum())
        st.markdown(
            f"- Postingan terindikasi radikal (rule ketat): **{n_rad_rule}**"
        )
        st.markdown(
            f"- Postingan terindikasi radikal (skor final rule+LLM ≥ 0.7): **{n_rad_final}**"
        )
        st.markdown(
            f"- Postingan anomali (engagement): **{(df['anomali_flag'] == 'Anomali').sum()}**"
        )
        st.markdown(f"- SNA feature source: **{sna_source}**")

# ------------------------------------------------------------
# TAB 2 — NLP & WORDCLOUD
# ------------------------------------------------------------
with tab2:
    st.header("NLP: Rule-based & WordCloud (Mode Ketat)")

    st.subheader("Distribusi Sentimen Rule-based (keyword)")
    sent_counts = df["sentimen_rule"].value_counts().reset_index()
    sent_counts.columns = ["sentimen", "count"]
    fig_sent = px.bar(
        sent_counts,
        x="sentimen",
        y="count",
        title="Distribusi Sentimen (rule-based sederhana)",
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    st.subheader("Distribusi Radikal (Rule Ketat)")
    rad_counts = df["indikator_radikal_rule"].value_counts().reset_index()
    rad_counts.columns = ["radikal", "count"]
    rad_counts["label"] = rad_counts["radikal"].map(
        {1: "Radikal (rule ketat)", 0: "Non-radikal"}
    )
    fig_rad = px.pie(
        rad_counts,
        names="label",
        values="count",
        title="Proporsi Postingan Radikal (rule ketat)",
    )
    st.plotly_chart(fig_rad, use_container_width=True)

    st.subheader("WordCloud — Semua Postingan")
    plot_wordcloud_from_series(
        df["clean_text"], title="WordCloud - Semua Postingan"
    )

    st.subheader("WordCloud — Postingan Radikal (rule ketat)")
    plot_wordcloud_from_series(
        df.loc[df["indikator_radikal_rule"] == 1, "clean_text"],
        title="WordCloud - Radikal (rule ketat)",
    )

    st.subheader("Contoh Postingan Terindikasi Radikal (rule ketat)")
    st.dataframe(
        df[df["indikator_radikal_rule"] == 1][
            [
                "user_id",
                "nama",
                "isi_postingan",
                "total_engagement",
            ]
        ].reset_index(drop=True)
    )

# ------------------------------------------------------------
# TAB 3 — ANOMALY
# ------------------------------------------------------------
with tab3:
    st.header("Anomaly Detection (berbasis Engagement)")
    st.markdown(
        "Menggunakan IsolationForest pada **total_engagement** "
        "(jumlah Like + Komentar + Share)."
    )

    st.subheader("Postingan Anomali (total_engagement tidak wajar)")
    st.dataframe(
        df[df["anomali_flag"] == "Anomali"][
            ["user_id", "nama", "isi_postingan", "total_engagement"]
        ].reset_index(drop=True)
    )

    st.subheader(
        "Histogram Engagement (warna: radikal_final vs non-radikal)"
    )
    df["kategori_radikal_hist"] = df["indikator_radikal_final"].map(
        {1: "Radikal (final)", 0: "Non-Radikal"}
    )
    fig_hist = px.histogram(
        df,
        x="total_engagement",
        color="kategori_radikal_hist",
        nbins=30,
        title="Distribusi Total Engagement",
        color_discrete_map={
            "Radikal (final)": "red",
            "Non-Radikal": "blue",
        },
    )
    fig_hist.update_layout(
        bargap=0.05,
        xaxis_title="Total Engagement (Like + Komentar + Share)",
        yaxis_title="Jumlah Postingan",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------
# TAB 4 — LLM CLASSIFIER (INDOBERT)
# ------------------------------------------------------------
with tab4:
    st.header("LLM Classifier — IndoBERT Sentiment")

    if tokenizer_indobert is None:
        st.warning(
            "IndoBERT belum aktif (library transformers/torch belum terpasang "
            "atau model gagal di-load). Tampilkan hanya hasil rule-based."
        )

    st.subheader("Prediksi LLM per Postingan")
    st.dataframe(
        df[
            [
                "user_id",
                "nama",
                "isi_postingan",
                "indikator_radikal_rule",
                "llm_sentiment_label",
                "llm_sentiment_score",
                "llm_risk_score",
                "skor_radikal_final",
                "indikator_radikal_final",
            ]
        ].head(200)
    )

    if tokenizer_indobert is not None:
        st.subheader("Distribusi Label IndoBERT (Sentimen)")
        dist_llm = (
            df["llm_sentiment_label"]
            .value_counts(dropna=False)
            .reset_index()
        )
        dist_llm.columns = ["label", "count"]
        fig_llm = px.bar(
            dist_llm,
            x="label",
            y="count",
            title="Distribusi Label Sentimen (IndoBERT)",
        )
        st.plotly_chart(fig_llm, use_container_width=True)

    st.markdown(
        """
        **Catatan Metodologis singkat untuk penguji:**
        - IndoBERT yang digunakan adalah *sentiment classifier* publik dari HuggingFace
          (`w11wo/indonesian-roberta-base-sentiment-classifier`).
        - Output LLM digunakan sebagai **faktor risiko tambahan** (negatif = risiko lebih tinggi),
          lalu dikombinasikan dengan indikator rule-based (lexicon ketat) menjadi `skor_radikal_final`.
        - Threshold 0.7 digunakan untuk menetapkan `indikator_radikal_final`.
        """
    )


# ------------------------------------------------------------
# TAB 5 — SNA (Radikal saja) + LEGEND KLUSTER
# ------------------------------------------------------------
with tab5:
    st.header(
        "Social Network Analysis (SNA) — Akun dengan Konten Radikal (Final)"
    )
    st.markdown(
        f"""
        Graph dibangun dari postingan dengan **indikator_radikal_final = 1**.

        - Feature teks untuk similarity: **{sna_source}**
        - Node = akun (`user_id`)
        - Edge = kemiripan narasi (cosine similarity ≥ threshold)
        - Ukuran node ≈ jumlah posting radikal
        - Warna node = **kluster narasi internal** (berbasis embedding teks)
        """
    )

    df_rad = df[df["indikator_radikal_final"] == 1].copy()

    if len(df_rad) < 2:
        st.info("Belum cukup data untuk membangun graph SNA.")
    else:
        # ----------------------------------------------------
        # DATA SNA
        # ----------------------------------------------------
        rad_idx = df_rad.index.tolist()
        feature_rad = feature_matrix_sna[rad_idx, :]
        df_posts_rad = df_rad[["user_id", "nama"]].reset_index(drop=True)

        G = build_similarity_graph(
            feature_rad, df_posts_rad, threshold=sim_threshold
        )

        if G.number_of_nodes() == 0:
            st.info("Tidak ada edge pada threshold saat ini.")
        else:
            # ------------------------------------------------
            # KLUSTERISASI NARASI (SATU KALI)
            # ------------------------------------------------
            account_clusters = cluster_accounts_by_narrative(
                feature_rad,
                df_posts_rad,
                n_clusters=3
            )

            # PALET WARNA + LABEL LEGEND
            cluster_info = {
                0: {"color": "red",  "label": "Ideologis / Doktrinal"},
                1: {"color": "blue", "label": "Kekerasan / Jihad"},
                2: {"color": "gold", "label": "Rekrutmen / Hijrah"},
            }

            pos = nx.spring_layout(G, seed=42, k=0.6)

            # ------------------------------------------------
            # EDGE TRACE
            # ------------------------------------------------
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.7, color="#888"),
                hoverinfo="none",
                showlegend=False,
            )

            # ------------------------------------------------
            # NODE TRACE PER KLUSTER (AGAR ADA LEGEND)
            # ------------------------------------------------
            node_traces = []

            for cid, info in cluster_info.items():
                nx_, ny_, nt_, ns_ = [], [], [], []

                for n, d in G.nodes(data=True):
                    if account_clusters.get(n) != cid:
                        continue

                    x, y = pos[n]
                    nx_.append(x)
                    ny_.append(y)

                    ns_.append(node_size_base + d.get("posts", 0) * 5)

                    nt_.append(
                        f"{d.get('label','')}"
                        f"<br>UID: {n}"
                        f"<br>Posting radikal: {d.get('posts',0)}"
                        f"<br>Kluster: {info['label']}"
                    )

                node_traces.append(
                    go.Scatter(
                        x=nx_,
                        y=ny_,
                        mode="markers+text",
                        name=info["label"],  # INI LEGEND
                        text=[G.nodes[n].get("label","") for n in G.nodes()
                              if account_clusters.get(n) == cid],
                        textposition="top center",
                        hovertext=nt_,
                        marker=dict(
                            size=ns_,
                            color=info["color"],
                            line=dict(width=1, color="#333"),
                        ),
                    )
                )

            # ------------------------------------------------
            # PLOT FINAL
            # ------------------------------------------------
            fig_sna = go.Figure(
                data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=(
                        f"SNA Akun Radikal — Kluster Narasi Internal<br>"
                        f"<sup>Sumber fitur: {sna_source}, threshold cosine = {sim_threshold:.2f}</sup>"
                    ),
                    showlegend=True,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )

            st.plotly_chart(fig_sna, use_container_width=True)

            # ------------------------------------------------
            # DEGREE CENTRALITY
            # ------------------------------------------------
            st.subheader("Akun dengan koneksi terbanyak di jaringan radikal")
            top_deg = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]

            rows = []
            for uid, deg in top_deg:
                rows.append(
                    {
                        "user_id": uid,
                        "nama": G.nodes[uid].get("label", ""),
                        "degree": deg,
                        "jml_post_radikal": G.nodes[uid].get("posts", 0),
                        "kluster_narasi": cluster_info.get(
                            account_clusters.get(uid), {}
                        ).get("label", "Tidak diketahui"),
                    }
                )

            st.table(pd.DataFrame(rows))



# ------------------------------------------------------------
# TAB 6 — REKOMENDASI CEGAH DINI
# ------------------------------------------------------------
with tab6:
    st.header("Rekomendasi Cegah Dini (berdasarkan skor final)")

    n_radikal = int(df["indikator_radikal_final"].sum())
    n_anomali_total = int((df["anomali_flag"] == "Anomali").sum())
    n_anomali_radikal = int(
        (
            (df["anomali_flag"] == "Anomali")
            & (df["indikator_radikal_final"] == 1)
        ).sum()
    )

    st.subheader("Ringkasan Kuantitatif")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Postingan Radikal (skor final)", n_radikal)
    with col_b:
        st.metric("Postingan Anomali (engagement)", n_anomali_total)
    with col_c:
        st.metric("Anomali + Radikal", n_anomali_radikal)

    st.subheader("Akun dengan Indikasi Risiko Tinggi (berdasarkan skor final)")
    top_rad_user = (
        df[df["indikator_radikal_final"] == 1]
        .groupby(["user_id", "nama"])
        .size()
        .reset_index(name="jumlah_post_radikal")
        .sort_values("jumlah_post_radikal", ascending=False)
        .head(5)
    )

    top_eng_rad = (
        df[df["indikator_radikal_final"] == 1]
        .sort_values("total_engagement", ascending=False)[
            ["user_id", "nama", "isi_postingan", "total_engagement"]
        ]
        .head(5)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Akun dengan jumlah posting radikal (skor final) terbanyak**"
        )
        if len(top_rad_user) > 0:
            st.table(top_rad_user)
        else:
            st.info("Belum ada akun dengan posting radikal (skor final).")

    with col2:
        st.markdown(
            "**Postingan radikal dengan engagement tertinggi (skor final)**"
        )
        if len(top_eng_rad) > 0:
            st.dataframe(top_eng_rad.reset_index(drop=True))
        else:
            st.info(
                "Belum ada postingan radikal dengan engagement signifikan."
            )

    st.subheader("Saran Tindak Lanjut (Heuristik Analitik)")

    st.markdown(
        """
        **1. Fokus pemantauan konten dan akun prioritas**
        - Prioritaskan penelaahan manual terhadap:
          - Akun dengan **jumlah posting radikal (skor final) terbanyak**.
          - Postingan radikal dengan **engagement paling tinggi**.
          - Node dengan **degree** tertinggi pada grafik SNA.
        - Lakukan verifikasi konteks teks:
          - Apakah benar berisi ajakan kekerasan / glorifikasi kelompok teroris,
            atau sekadar diskusi, kritik, maupun edukasi anti-radikalisme.

        **2. Pemantauan sebaran (Propagation Risk)**
        - Gunakan kombinasi:
          - Hasil **IsolationForest** (engagement tidak wajar).
          - Struktur jaringan **SNA** (berbasis MiniLM/TF-IDF) untuk mengidentifikasi
            akun yang berperan sebagai **hub pengaruh**.

        **3. Penguatan kontra narasi & intervensi lunak**
        - Dorong konten tandingan:
          - Narasi damai, toleransi, dan penolakan kekerasan.
          - Penjelasan keagamaan moderat atas istilah yang sering diselewengkan.
        - Kolaborasi dengan tokoh agama moderat, komunitas lokal, dan lembaga pendidikan.

        **4. Pengembangan sistem**
        - Lanjutkan penyusunan **dataset berlabel manual (ground truth)** agar
          IndoBERT bisa di-finetune khusus untuk label *pro-radikal vs kontra-radikal*.
        - Integrasi dimensi waktu (timestamp) untuk menganalisis **eskalasi tren**.
        - Evaluasi berkala lexicon dan aturan mode ketat berdasarkan temuan baru di lapangan.

        """
    )

st.success("Analisis selesai (Mode 3: Rule-based + IndoBERT + MiniLM).")
