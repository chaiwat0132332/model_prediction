# ==========================================================
# 🤖 ระบบพยากรณ์ AI (เวอร์ชันใช้งานจริง)
# คุณสมบัติ:
# - สอนโมเดล / พยากรณ์อนาคต
# - ตรวจสอบข้อมูลก่อนพยากรณ์
# - วิเคราะห์ความน่าเชื่อถือและให้คำแนะนำ
# - ส่งออกข้อมูลเป็น Excel และ CSV
# - ลบโมเดลที่บันทึกไว้
# - แสดงความคืบหน้าและกราฟการเรียนรู้ระหว่างเทรน
# ==========================================================

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================================
# เชื่อมต่อระบบหลังบ้าน (BACKEND)
# ==========================================================

try:
    from src.train.pipeline import run_training
    from src.utils.model_io import list_models, load_model, save_model, delete_model
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในระบบหลังบ้าน: {e}")
    st.stop()


# ==========================================================
# การตั้งค่าหน้าจอ
# ==========================================================

st.set_page_config(
    layout="wide",
    page_title="ระบบพยากรณ์ AI อัจฉริยะ",
    page_icon="🤖",
)

st.title("🤖 ระบบพยากรณ์ AI (Smart Forecast)")
st.caption("เครื่องมือวิเคราะห์และทำนายข้อมูลด้วยเทคโนโลยี LSTM และ Linear Regression")


# ==========================================================
# ฟังก์ชันคำนวณความสัมพันธ์ของข้อมูล
# ==========================================================

def autocorr(x, lag=1):
    x = np.asarray(x, dtype=float)

    if len(x) <= lag:
        return 0.0

    a = x[:-lag]
    b = x[lag:]

    if len(a) == 0 or len(b) == 0:
        return 0.0

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    value = np.corrcoef(a, b)[0, 1]
    if np.isnan(value):
        return 0.0

    return float(value)


# ==========================================================
# ฟังก์ชันทำความสะอาดข้อมูลด้วยมือ
# ==========================================================

def manual_clean_data(df, col, z_threshold, window_size):
    df_out = df.copy()

    series = pd.to_numeric(df_out[col], errors="coerce")
    z = np.abs(stats.zscore(series, nan_policy="omit"))
    z = np.nan_to_num(z)
    outliers = z > z_threshold

    if outliers.sum() > 0:
        median = series.median()
        df_out.loc[outliers, col] = median

    if window_size > 1:
        df_out[col] = (
            pd.to_numeric(df_out[col], errors="coerce")
            .rolling(window_size, center=True)
            .mean()
            .bfill()
            .ffill()
        )

    return df_out, int(outliers.sum())


# ==========================================================
# ฟังก์ชันโหลดข้อมูล
# ==========================================================

def load_data():
    file = st.file_uploader(
        "📁 อัปโหลดไฟล์ CSV หรือ Excel",
        type=["csv", "xlsx"],
    )
    if file is None:
        st.stop()

    try:
        if file.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(file)
        else:
            df_raw = pd.read_excel(file)
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ได้: {e}")
        st.stop()

    if df_raw.empty:
        st.error("ไฟล์ว่าง หรือไม่มีข้อมูลที่อ่านได้")
        st.stop()

    if "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(
            df_raw["Date"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
        df_raw = df_raw.dropna(subset=["Date"])

    st.subheader("📊 ตรวจสอบความสมบูรณ์ของไฟล์")

    m1, m2, m3 = st.columns(3)
    total_rows = len(df_raw)
    total_nan = int(df_raw.isnull().sum().sum())

    m1.metric("จำนวนแถวทั้งหมด", f"{total_rows:,}")
    m2.metric(
        "ค่าที่หายไป (NaN) ทั้งไฟล์",
        f"{total_nan:,}",
        delta=f"{total_nan}" if total_nan > 0 else None,
        delta_color="inverse",
    )
    m3.metric("จำนวนคอลัมน์", f"{len(df_raw.columns)}")

    with st.expander("ดูตัวอย่างข้อมูล 10 แถวแรก", expanded=True):
        st.dataframe(df_raw.head(10), use_container_width=True)

        nan_info = df_raw.isnull().sum()
        if nan_info.sum() > 0:
            st.warning("⚠️ ตรวจพบค่าว่างในคอลัมน์:")
            st.write(nan_info[nan_info > 0])
        else:
            st.success("✅ ข้อมูลสมบูรณ์ (ไม่พบค่าว่าง)")

    st.divider()

    target = st.selectbox("เลือกคอลัมน์ที่ต้องการทำนาย (Target)", df_raw.columns)

    series = pd.to_numeric(df_raw[target], errors="coerce")
    nan_in_target = int(series.isna().sum())

    if nan_in_target > 0:
        st.info(
            f"💡 ระบบพบค่าที่ไม่ใช่ตัวเลขหรือค่าว่างในคอลัมน์ '{target}' "
            f"จำนวน {nan_in_target} แถว (จะถูกตัดออกเพื่อใช้ในการประมวลผล)"
        )

    series = series.dropna().reset_index(drop=True)

    if len(series) < 3:
        st.error("ข้อมูลเชิงตัวเลขในคอลัมน์ที่เลือกมีน้อยเกินไป")
        st.stop()

    df_out = pd.DataFrame(
        {
            "value": series,
            "time": np.arange(len(series)),
        }
    )

    return df_out, "time", "value"


# ==========================================================
# ฟังก์ชันสร้างกราฟ learning curve
# ==========================================================

def render_learning_curve(train_loss, val_loss, current_epoch=None, total_epochs=None):
    fig_lc = go.Figure()

    fig_lc.add_trace(
        go.Scatter(
            x=list(range(1, len(train_loss) + 1)),
            y=train_loss,
            name="Train Loss",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig_lc.add_trace(
        go.Scatter(
            x=list(range(1, len(val_loss) + 1)),
            y=val_loss,
            name="Validation Loss",
            line=dict(color="#ff7f0e", width=2),
        )
    )

    title = "📉 กราฟการเรียนรู้ (Learning Curve)"
    if current_epoch is not None and total_epochs is not None:
        title = f"📉 กราฟการเรียนรู้ระหว่างเทรน (Epoch {current_epoch}/{total_epochs})"

    fig_lc.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=400,
    )
    return fig_lc


# ==========================================================
# แถบเมนูข้าง (SIDEBAR)
# ==========================================================

with st.sidebar:
    st.header("เมนูหลัก")
    mode = st.radio("โหมดการทำงาน", ["สอนโมเดล (Train)", "พยากรณ์ (Forecast)"])
    st.divider()
    st.caption("เวอร์ชัน: Production 2.2")


# ==========================================================
# โหมดสอนโมเดล (TRAIN MODE)
# ==========================================================

if mode == "สอนโมเดล (Train)":
    st.header("สอนโมเดล (Train Model)")

    df, time_col, target_col = load_data()

    col1, col2 = st.columns(2)
    with col1:
        z = st.slider(
            "ระดับการกำจัดค่าผิดปกติ (Outlier Z-score)",
            1.0,
            5.0,
            3.0,
            help="ค่าน้อยจะกำจัดค่าที่กระโดดออกจากกลุ่มมาก",
        )
    with col2:
        smooth = 50

    df_clean, out_count = manual_clean_data(df, target_col, z, smooth)
    st.info(f"🔍 ตรวจพบค่าผิดปกติ: {out_count} จุด (ถูกแทนที่ด้วยค่ากลางแล้ว)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[target_col],
            name="ข้อมูลเดิม",
            line=dict(color="silver"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_clean[time_col],
            y=df_clean[target_col],
            name="ข้อมูลที่คลีนแล้ว",
            line=dict(color="#1f77b4"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="การเปรียบเทียบข้อมูลก่อนและหลังการเตรียมการ",
    )
    st.plotly_chart(fig, use_container_width=True)

    ac = autocorr(df_clean[target_col].values)
    st.metric("ความสัมพันธ์รายเวลา (Autocorrelation)", f"{ac:.3f}")
    if ac < 0.3:
        st.warning("⚠️ ข้อมูลมีความสัมพันธ์รายเวลาต่ำ โมเดลอาจพยากรณ์ได้ไม่แม่นยำนัก")

    st.subheader("⚙️ ตั้งค่าโมเดล")
    model_type = st.selectbox(
        "เลือกประเภทโมเดล",
        ["linear", "lstm"],
        format_func=lambda x: "Linear Regression" if x == "linear" else "LSTM (Deep Learning)",
    )

    max_lag = min(200, max(1, len(df_clean) - 1))
    default_lag = min(5, max_lag)

    lag = st.number_input(
        "จำนวนข้อมูลย้อนหลังที่ใช้ทาย (Lag)",
        min_value=1,
        max_value=max_lag,
        value=default_lag,
    )

    if model_type == "lstm":
        c1, c2, c3 = st.columns(3)
        epochs = int(c1.number_input("รอบการสอน (Epochs)", 10, 500, 100))
        hidden = int(c2.number_input("ขนาดความจำ (Hidden size)", 16, 512, 128))
        dropout = float(c3.slider("Dropout", 0.0, 0.5, 0.2))
    else:
        epochs = None
        hidden = None
        dropout = None

    model_name = st.text_input("ชื่อโมเดล", f"model_{datetime.now().strftime('%H%M%S')}")

    if len(df_clean) <= lag:
        st.error("ข้อมูลน้อยกว่าค่า lag ที่ตั้งไว้")
        st.stop()

    if st.button("🚀 เริ่มสอนโมเดล"):
        progress_bar = st.progress(0)
        status_box = st.empty()
        epoch_box = st.empty()
        chart_box = st.empty()

        live_train_loss = []
        live_val_loss = []

        def progress_callback(current_epoch, total_epochs, train_loss, val_loss):
            live_train_loss.append(float(train_loss))
            live_val_loss.append(float(val_loss))

            percent = max(0, min(100, int(current_epoch / total_epochs * 100)))
            progress_bar.progress(percent)
            status_box.info(f"🧠 AI กำลังเรียนรู้ข้อมูล... Epoch {current_epoch}/{total_epochs}")
            epoch_box.metric("รอบการเทรนปัจจุบัน", f"{current_epoch}/{total_epochs}")

            fig_live = render_learning_curve(
                live_train_loss,
                live_val_loss,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
            )
            chart_box.plotly_chart(fig_live, use_container_width=True)

        try:
            run_kwargs = dict(
                df=df_clean,
                target_col=target_col,
                model_type=model_type,
                lag=lag,
                hidden_size=hidden,
                num_layers=2,
                dropout=dropout,
                epochs=epochs,
                forecast_horizon=120,
            )

            if model_type == "lstm":
                try:
                    artifact = run_training(
                        progress_callback=progress_callback,
                        **run_kwargs,
                    )
                except TypeError:
                    status_box.warning("backend ยังไม่รองรับ progress_callback จะแสดงผลหลังเทรนเสร็จแทน")
                    artifact = run_training(**run_kwargs)
            else:
                artifact = run_training(**run_kwargs)

            progress_bar.progress(100)
            status_box.success("✅ เทรนโมเดลเสร็จแล้ว")

            save_model(artifact, model_name)

            r2 = r2_score(artifact["test_true"], artifact["test_pred"])
            mse = mean_squared_error(artifact["test_true"], artifact["test_pred"])

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("ความแม่นยำ (R²)", f"{r2:.4f}")
            col_m2.metric("ความคลาดเคลื่อน (MSE)", f"{mse:.4f}")

            learning_curve = artifact.get("learning_curve")
            if learning_curve:
                train_loss = learning_curve.get("train_loss", [])
                val_loss = learning_curve.get("val_loss", [])

                if len(train_loss) > 0 and len(val_loss) > 0:
                    st.subheader("📉 กราฟการเรียนรู้ (Learning Curve)")
                    final_fig = render_learning_curve(train_loss, val_loss)
                    st.plotly_chart(final_fig, use_container_width=True)

                    with st.expander("💡 วิธีการอ่านกราฟ Learning Curve", expanded=False):
                        st.markdown(
                            """
- **กราฟลดลงทั้งคู่:** โมเดลเรียนรู้ได้ดี
- **Train ลดแต่ Validation พุ่งขึ้น:** เกิด **Overfitting**
- **กราฟนิ่งทั้งที่ Loss ยังสูง:** เกิด **Underfitting**
                            """
                        )

            st.subheader("💡 การวินิจฉัยและคำแนะนำจาก AI")
            if r2 < 0.3:
                st.error("❌ **ผลลัพธ์: ต่ำมาก**")
                st.markdown(
                    """
**คำแนะนำเพื่อปรับปรุง:**
1. เพิ่มค่า Lag
2. ตรวจสอบว่าข้อมูลมีรูปแบบจริงหรือไม่
3. หากใช้ Linear ลองเปลี่ยนเป็น LSTM หรือเพิ่ม Epochs
                    """
                )
            elif r2 < 0.6:
                st.warning("⚠️ **ผลลัพธ์: ปานกลาง**")
                st.markdown(
                    """
**คำแนะนำเพื่อปรับปรุง:**
1. ปรับความสมูท
2. เพิ่ม Hidden Size ใน LSTM
                    """
                )
            else:
                st.success("✅ **ผลลัพธ์: ดีมาก**")
                st.write("โมเดลเรียนรู้รูปแบบข้อมูลได้ดีเยี่ยม พร้อมใช้งานพยากรณ์แล้ว!")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่างการสอนโมเดล: {e}")


# ==========================================================
# โหมดพยากรณ์ (FORECAST MODE)
# ==========================================================

elif mode == "พยากรณ์ (Forecast)":
    st.header("🔮 การพยากรณ์ (Forecast)")

    models = list_models()
    if not models:
        st.warning("⚠️ ไม่พบโมเดลในระบบ โปรดไปที่โหมด 'สอนโมเดล' ก่อน")
        st.stop()

    col_sel, col_del = st.columns([3, 1])

    with col_sel:
        model_sel = st.selectbox("🎯 เลือกโมเดลที่จะใช้พยากรณ์", models)

    with col_del:
        st.write("")
        st.write("")
        if st.button("🗑️ ลบโมเดล", use_container_width=True):
            try:
                delete_model(model_sel)
                st.success(f"ลบโมเดล '{model_sel}' แล้ว")
                st.rerun()
            except Exception as e:
                st.error(f"ลบโมเดลไม่สำเร็จ: {e}")

    try:
        artifact = load_model(model_sel)
    except Exception as e:
        st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
        st.stop()

    config = artifact.get("config", {})
    lag = int(config["lag"])
    model_type = config.get("model_type", "N/A")

    st.info(f"📋 ข้อมูลโมเดล: ประเภท {str(model_type).upper()} | ค่า Lag ที่ใช้: {lag}")
    st.caption(f"Train Horizon: {config.get('forecast_horizon', 'N/A')}")

    col_m1, col_m2 = st.columns(2)
    metrics = artifact.get("metrics")
    if metrics:
        col_m1.metric("ความแม่นยำ (R²)", f"{metrics['r2']:.4f}")
        col_m2.metric("ความคลาดเคลื่อน (MSE)", f"{metrics['mse']:.4f}")
    else:
        st.warning("โมเดลนี้ไม่มีค่า metrics")

    df, time_col, target_col = load_data()

    st.subheader("🛠️ ปรับแต่งข้อมูลก่อนพยากรณ์")
    col1, col2 = st.columns(2)
    with col1:
        z_forecast = 3.0
    with col2:
        smooth_val = 50

    df_clean, out_count = manual_clean_data(df, target_col, z_forecast, smooth_val)

    if out_count > 0:
        st.info(f"🔍 ตรวจพบค่าผิดปกติในชุดข้อมูลนี้: {out_count} จุด (จัดการเรียบร้อยแล้ว)")

    if len(df_clean) < lag:
        st.error("ข้อมูลมีน้อยกว่าค่า Lag ที่ตั้งไว้")
        st.stop()

    fig_clean = go.Figure()
    fig_clean.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df[target_col],
            name="ข้อมูลดิบ",
            line=dict(color="silver"),
        )
    )
    fig_clean.add_trace(
        go.Scatter(
            x=df_clean[time_col],
            y=df_clean[target_col],
            name="ข้อมูลที่ปรับแต่งแล้ว",
            line=dict(color="#00CC96"),
        )
    )

    last_idx = df_clean[time_col].iloc[-lag:]
    last_val = df_clean[target_col].iloc[-lag:]
    fig_clean.add_trace(
        go.Scatter(
            x=last_idx,
            y=last_val,
            name="หน้าต่างข้อมูลที่ AI ใช้ทาย",
            line=dict(color="red", width=4),
        )
    )

    fig_clean.update_layout(
        template="plotly_white",
        title="การเตรียมข้อมูลฐานสำหรับการพยากรณ์",
    )
    st.plotly_chart(fig_clean, use_container_width=True)

    st.divider()
    horizon = int(
        st.number_input(
            "🚀 จำนวนก้าวที่ต้องการพยากรณ์ไปข้างหน้า",
            min_value=1,
            max_value=10000,
            value=24,
        )
    )

    if st.button("✨ เริ่มพยากรณ์อนาคต"):
        with st.spinner("🤖 AI กำลังประมวลผล..."):
            try:
                series = df_clean[target_col].values
                if len(series) < lag:
                    st.error("ข้อมูลมีน้อยกว่าค่า Lag ที่ตั้งไว้")
                    st.stop()

                last_window = series[-lag:]
                model = artifact["model"]

                future = model.forecast(last_window, steps=horizon)
                future = np.asarray(future, dtype=float).flatten()

                st.subheader("📈 ผลลัพธ์การพยากรณ์")

                last_time = int(df_clean[time_col].iloc[-1])
                future_x = np.arange(last_time + 1, last_time + 1 + len(future))

                hist_window = min(len(df_clean), lag * 2)

                fig_res = go.Figure()
                fig_res.add_trace(
                    go.Scatter(
                        x=df_clean[time_col].iloc[-hist_window:],
                        y=df_clean[target_col].iloc[-hist_window:],
                        name="ข้อมูลล่าสุด",
                    )
                )
                fig_res.add_trace(
                    go.Scatter(
                        x=future_x,
                        y=future,
                        name="ค่าพยากรณ์",
                        line=dict(color="red", width=3),
                    )
                )

                fig_res.update_layout(
                    template="plotly_white",
                    title=f"พยากรณ์ล่วงหน้า {horizon} ช่วงเวลา",
                )
                st.plotly_chart(fig_res, use_container_width=True)

                if model_type == "lstm":
                    st.write("is_fitted:", getattr(model, "is_fitted", None))
                    if hasattr(model, "model"):
                        st.write("device:", next(model.model.parameters()).device)

                result_df = pd.DataFrame(
                    {
                        "Time_Index": future_x.astype(int),
                        "Forecast_Value": future.astype(float),
                    }
                )

                col_d1, col_d2 = st.columns([2, 1])
                with col_d1:
                    st.dataframe(result_df, use_container_width=True)

                with col_d2:
                    st.write("💾 ดาวน์โหลดไฟล์")
                    st.download_button(
                        "📥 CSV",
                        result_df.to_csv(index=False).encode("utf-8-sig"),
                        "forecast.csv",
                        mime="text/csv",
                    )

                    output = io.BytesIO()
                    with pd.ExcelWriter(output) as writer:
                        result_df.to_excel(writer, index=False, sheet_name="forecast")

                    st.download_button(
                        "📥 Excel",
                        output.getvalue(),
                        "forecast.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างการพยากรณ์: {e}")
