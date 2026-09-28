from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="Model Health",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Model Health")
st.caption("حالة النموذج والبيانات والنتائج المحفوظة")

metadata_path = MODELS / "model_metadata.json"
feature_path = MODELS / "feature_order.json"
predictions_path = RESULTS / "predictions_test.csv"
analysis_path = RESULTS / "fault_attack_analysis.csv"

required_files = {
    "Random Forest": MODELS / "random_forest.joblib",
    "XGBoost": MODELS / "xgboost_model.joblib",
    "Imputer": MODELS / "imputer.joblib",
    "Label Encoder": MODELS / "label_encoder.joblib",
    "Feature Order": feature_path,
}

st.subheader("حالة الملفات")

status_rows = []

for name, path in required_files.items():
    status_rows.append(
        {
            "Component": name,
            "Status": "Available" if path.exists() else "Missing",
            "Path": str(path),
        }
    )

status_df = pd.DataFrame(status_rows)
st.dataframe(status_df, use_container_width=True, hide_index=True)

if metadata_path.exists():
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    st.subheader("معلومات التدريب")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Training Rows",
        f"{metadata.get('train_rows', 0):,}",
    )

    c2.metric(
        "Test Rows",
        f"{metadata.get('test_rows', 0):,}",
    )

    c3.metric(
        "Features",
        metadata.get("features", 0),
    )

    c4.metric(
        "Ignored Rows",
        f"{metadata.get('ignored_unknown_rows', 0):,}",
    )

    st.write("Target column:", metadata.get("target"))
    st.write("Classes:", ", ".join(metadata.get("classes", [])))

    with st.expander("تفاصيل Metadata"):
        st.json(metadata)

if predictions_path.exists():
    predictions = pd.read_csv(predictions_path)

    st.subheader("إحصاءات الاختبار")

    rf_accuracy = (
        predictions["y_true"]
        == predictions["rf_prediction"]
    ).mean()

    xgb_accuracy = (
        predictions["y_true"]
        == predictions["xgb_prediction"]
    ).mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Random Forest Accuracy",
        f"{rf_accuracy:.2%}",
    )

    c2.metric(
        "XGBoost Accuracy",
        f"{xgb_accuracy:.2%}",
    )

    c3.metric(
        "Test Samples",
        f"{len(predictions):,}",
    )

    st.subheader("توزيع الفئات الحقيقية")
    st.bar_chart(predictions["y_true"].value_counts())

if analysis_path.exists():
    st.subheader("الخطأ الأمني الحرج")

    analysis = pd.read_csv(analysis_path)
    st.dataframe(
        analysis,
        use_container_width=True,
        hide_index=True,
    )

st.info(
    "هذه الصفحة للتدقيق الأكاديمي ومراقبة حالة الملفات والنتائج. "
    "لا تمثل اتصالاً بنظام كهرباء حقيقي."
)