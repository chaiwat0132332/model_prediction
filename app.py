# ==========================================================
# 🤖 ระบบพยากรณ์ AI (เวอร์ชันใช้งานจริง)
# คุณสมบัติ:
# - สอนโมเดล / พยากรณ์อนาคต
# - ตรวจสอบข้อมูลก่อนพยากรณ์
# - วิเคราะห์ความน่าเชื่อถือและให้คำแนะนำ
# - ส่งออกข้อมูลเป็น Excel และ CSV
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import io


# ==========================================================
# เชื่อมต่อระบบหลังบ้าน (BACKEND)
# ==========================================================

try:
    from src.train.pipeline import run_training
    from src.utils.model_io import save_model, load_model, list_models
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในระบบหลังบ้าน: {e}")
    st.stop()


# ==========================================================
# การตั้งค่าหน้าจอ
# ==========================================================

st.set_page_config(
    layout="wide",
    page_title="ระบบพยากรณ์ AI อัจฉริยะ",
    page_icon="🤖"
)

st.title("🤖 ระบบพยากรณ์ AI (Smart Forecast)")
st.caption("เครื่องมือวิเคราะห์และทำนายข้อมูลด้วยเทคโนโลยี LSTM และ Linear Regression")


# ==========================================================
# ฟังก์ชันคำนวณความสัมพันธ์ของข้อมูล
# ==========================================================

def autocorr(x, lag=1):
    if len(x) <= lag:
        return 0
    return np.corrcoef(x[:-lag], x[lag:])[0, 1]


# ==========================================================
# ฟังก์ชันทำความสะอาดข้อมูลด้วยมือ
# ==========================================================

def manual_clean_data(df, col, z_threshold, window_size):
    df_out = df.copy()
    z = np.abs(stats.zscore(df_out[col]))
    outliers = z > z_threshold

    if outliers.sum() > 0:
        median = df_out[col].median()
        df_out.loc[outliers, col] = median

    if window_size > 1:
        df_out[col] = (
            df_out[col]
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
        type=["csv", "xlsx"]
    )
    if file is None:
        st.stop()

    # โหลดไฟล์ต้นฉบับเพื่อตรวจสอบเบื้องต้น
    df_raw = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    
    if "Date" in df_raw.columns:
        df_raw["Date"] = pd.to_datetime(
            df_raw["Date"].astype(str),
            errors="coerce",
            dayfirst=True
        )
    # --- ส่วนการพรีวิวและแจ้งเตือนสถิติข้อมูล ---
    st.subheader("📊 ตรวจสอบความสมบูรณ์ของไฟล์")
    
    # 1. แสดง Metrics หลัก
    m1, m2, m3 = st.columns(3)
    total_rows = len(df_raw)
    total_nan = df_raw.isnull().sum().sum()
    
    m1.metric("จำนวนแถวทั้งหมด", f"{total_rows:,}")
    m2.metric("ค่าที่หายไป (NaN) ทั้งไฟล์", f"{total_nan:,}", delta=f"{total_nan}" if total_nan > 0 else None, delta_color="inverse")
    m3.metric("จำนวนคอลัมน์", f"{len(df_raw.columns)}")

    # 2. พรีวิวข้อมูล 10 แถวแรก
    with st.expander(" ดูตัวอย่างข้อมูล 10 แถวแรก", expanded=True):
        st.dataframe(df_raw.head(10), width="stretch")
        
        # แสดงรายการ NaN รายคอลัมน์ (ถ้ามี)
        nan_info = df_raw.isnull().sum()
        if nan_info.sum() > 0:
            st.warning("⚠️ ตรวจพบค่าว่างในคอลัมน์:")
            # กรองเฉพาะคอลัมน์ที่มี NaN
            st.write(nan_info[nan_info > 0])
        else:
            st.success("✅ ข้อมูลสมบูรณ์ (ไม่พบค่าว่าง)")

    st.divider()

    # 3. เลือกคอลัมน์เป้าหมาย
    target = st.selectbox(" เลือกคอลัมน์ที่ต้องการทำนาย (Target)", df_raw.columns)

    # จัดการข้อมูล: แปลงเป็นตัวเลข และลบ NaN เฉพาะส่วนที่จำเป็น
    series = pd.to_numeric(df_raw[target], errors="coerce")
    nan_in_target = series.isna().sum()
    
    if nan_in_target > 0:
        st.info(f"💡 ระบบพบค่าที่ไม่ใช่ตัวเลขหรือค่าว่างในคอลัมน์ '{target}' จำนวน {nan_in_target} แถว (จะถูกตัดออกเพื่อใช้ในการประมวลผล)")

    series = series.dropna().reset_index(drop=True)
    df_out = pd.DataFrame({
        "value": series,
        "time": np.arange(len(series))
    })

    return df_out, "time", "value"


# ==========================================================
# แถบเมนูข้าง (SIDEBAR)
# ==========================================================

with st.sidebar:
    st.header(" เมนูหลัก")
    mode = st.radio("โหมดการทำงาน", [" สอนโมเดล (Train)", " พยากรณ์ (Forecast)"])
    st.divider()
    st.caption("เวอร์ชัน: Production 2.0")


# ==========================================================
# โหมดสอนโมเดล (TRAIN MODE)
# ==========================================================

if mode == " สอนโมเดล (Train)":
    st.header(" สอนโมเดล (Train Model)")

    df, time_col, target_col = load_data()

    col1, col2 = st.columns(2)
    with col1:
        z = st.slider("ระดับการกำจัดค่าผิดปกติ (Outlier Z-score)", 1.0, 5.0, 3.0, help="ค่าน้อยจะกำจัดค่าที่กระโดดออกจากกลุ่มมาก")
    with col2:
        smooth = 50

    df_clean, out_count = manual_clean_data(df, target_col, z, smooth)
    st.info(f"🔍 ตรวจพบค่าผิดปกติ: {out_count} จุด (ถูกแทนที่ด้วยค่ากลางแล้ว)")

    # กราฟเปรียบเทียบ
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[target_col], name="ข้อมูลเดิม", line=dict(color='silver')))
    fig.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean[target_col], name="ข้อมูลที่คลีนแล้ว", line=dict(color='#1f77b4')))
    fig.update_layout(template="plotly_white", title="การเปรียบเทียบข้อมูลก่อนและหลังการเตรียมการ")
    st.plotly_chart(fig, width="stretch")

    # วิเคราะห์ความสัมพันธ์
    ac = autocorr(df_clean[target_col].values)
    st.metric("ความสัมพันธ์รายเวลา (Autocorrelation)", f"{ac:.3f}")
    if ac < 0.3:
        st.warning("⚠️ ข้อมูลมีความสัมพันธ์รายเวลาต่ำ โมเดลอาจพยากรณ์ได้ไม่แม่นยำนัก")

    # ตั้งค่าการสอน
    st.subheader("⚙️ ตั้งค่าโมเดล")
    model_type = st.selectbox("เลือกประเภทโมเดล", ["linear", "lstm"], format_func=lambda x: "Linear Regression" if x=="linear" else "LSTM (Deep Learning)")
    max_lag = max(2, len(df)-1)
    default_lag = min(5, max_lag)
    lag = st.number_input(
        "จำนวนข้อมูลย้อนหลังที่ใช้ทาย (Lag)",
        min_value=1,
        max_value=max_lag,
         value=default_lag
    )
    if model_type == "lstm":
        c1, c2, c3 = st.columns(3)
        epochs = c1.number_input("รอบการสอน (Epochs)", 10, 500, 100)
        hidden = c2.number_input("ขนาดความจำ (Hidden size)", 16, 512, 128)
        dropout = c3.slider("Dropout", 0.0, 0.5, 0.2)
    else:
        epochs = hidden = dropout = None

    model_name = st.text_input("ชื่อโมเดล", f"model_{datetime.now().strftime('%H%M%S')}")

    if st.button("🚀 เริ่มสอนโมเดล"):
        with st.spinner("🧠 AI กำลังเรียนรู้ข้อมูล..."):
            artifact = run_training(
            df_clean,
            target_col,
            model_type,
            lag,
            hidden_size=hidden,
            num_layers=2,
            dropout=dropout,
            epochs=epochs,
            forecast_horizon=120
        )
            save_model(artifact, model_name)

            r2 = r2_score(artifact["test_true"], artifact["test_pred"])
            mse = mean_squared_error(artifact["test_true"], artifact["test_pred"])

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("ความแม่นยำ (R²)", f"{r2:.4f}")
            col_m2.metric("ความคลาดเคลื่อน (MSE)", f"{mse:.4f}")

            if "learning_curve" in artifact:
                st.subheader("📉 กราฟการเรียนรู้ (Learning Curve)")
                lc_data = artifact["learning_curve"]
    
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(y=lc_data["train_loss"], name="Train Loss", line=dict(color='#1f77b4', width=2)))
                fig_lc.add_trace(go.Scatter(y=lc_data["val_loss"], name="Validation Loss", line=dict(color='#ff7f0e', width=2)))
    
                fig_lc.update_layout(
                    template="plotly_white",
                    xaxis_title="Epochs",
                    yaxis_title="Loss (MSE)",
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=400
                )
                st.plotly_chart(fig_lc, use_container_width=True)

    # คำแนะนำในการอ่านกราฟ
                with st.expander("💡 วิธีการอ่านกราฟ Learning Curve", expanded=False):
                    st.markdown("""
        * **กราฟลดลงทั้งคู่:** โมเดลเรียนรู้ได้ดี
        * **Train ลดแต่ Validation พุ่งขึ้น:** เกิด **Overfitting** (โมเดลจำแม่นเกินไปแต่ทายอนาคตไม่ได้) -> ควรเพิ่ม Dropout หรือลด Epochs
        * **กราฟนิ่งทั้งที่ Loss ยังสูง:** เกิด **Underfitting** -> ควรเพิ่ม Hidden Size หรือเพิ่มค่า Lag
        """)

            # --- ระบบวิเคราะห์และให้คำแนะนำ ---
            st.subheader("💡 การวินิจฉัยและคำแนะนำจาก AI")
            if r2 < 0.3:
                st.error("❌ **ผลลัพธ์: ต่ำมาก**")
                st.markdown("""
                **คำแนะนำเพื่อปรับปรุง:**
                1. **เพิ่มค่า Lag:** ลองเพิ่มจำนวนข้อมูลย้อนหลังเพื่อให้โมเดลเห็นรูปแบบที่กว้างขึ้น
                2. **ตรวจสอบข้อมูล:** ข้อมูลอาจมีความเป็นสุ่ม (Random) มากเกินไป หรือไม่มีรูปแบบที่ชัดเจน
                3. **เปลี่ยนประเภทโมเดล:** หากใช้ Linear ลองเปลี่ยนเป็น LSTM หรือเพิ่มค่า Epochs ใน LSTM
                """)
            elif r2 < 0.6:
                st.warning("⚠️ **ผลลัพธ์: ปานกลาง**")
                st.markdown("""
                **คำแนะนำเพื่อปรับปรุง:**
                1. **ปรับความสมูท:** ลองเพิ่ม/ลดค่า Smooth window ในขั้นตอนเตรียมข้อมูล
                2. **เพิ่มความจำ:** ลองเพิ่มค่า Hidden Size ใน LSTM เพื่อให้โมเดลจำรายละเอียดได้มากขึ้น
                """)
            else:
                st.success("✅ **ผลลัพธ์: ดีมาก**")
                st.write("โมเดลเรียนรู้รูปแบบข้อมูลได้ดีเยี่ยม พร้อมใช้งานพยากรณ์แล้ว!")


# ==========================================================
# โหมดพยากรณ์ (FORECAST MODE)
# ==========================================================
elif mode == " พยากรณ์ (Forecast)":
    st.header("🔮 การพยากรณ์ (Forecast)")

    # 1. โหลดโมเดล
    models = list_models()
    if not models:
        st.warning("⚠️ ไม่พบโมเดลในระบบ โปรดไปที่โหมด 'สอนโมเดล' ก่อน")
        st.stop()

    model_sel = st.selectbox("🎯 เลือกโมเดลที่จะใช้พยากรณ์", models)
    artifact = load_model(model_sel)
    lag = artifact["config"]["lag"]
    model_type = artifact["config"].get("model_type", "N/A")

    st.info(f"📋 ข้อมูลโมเดล: ประเภท {model_type.upper()} | ค่า Lag ที่ใช้: {lag}")
    st.caption(
    f"Train Horizon: {artifact['config']['forecast_horizon']}"
)
    col_m1, col_m2 = st.columns(2)
    metrics = artifact.get("metrics")
    if metrics:
        col_m1.metric("ความแม่นยำ (R²)", f"{metrics['r2']:.4f}")
        col_m2.metric("ความคลาดเคลื่อน (MSE)", f"{metrics['mse']:.4f}")
    else:
        st.warning("โมเดลนี้ไม่มีค่า metrics")

    # 2. โหลดข้อมูลใหม่สำหรับใช้เป็นฐานการพยากรณ์
    df, time_col, target_col = load_data()

    # 3. ส่วนการปรับแต่งข้อมูล (เหมือนหน้า Train)
    st.subheader("🛠️ ปรับแต่งข้อมูลก่อนพยากรณ์")
    col1, col2 = st.columns(2)
    with col1:
        z_forecast = 3.0
    with col2:
        smooth_val = 50

    # ใช้ฟังก์ชัน clean เดียวกับหน้าเทรน
    df_clean, out_count = manual_clean_data(df, target_col, z_forecast, smooth_val)
    
    if out_count > 0:
        st.info(f"🔍 ตรวจพบค่าผิดปกติในชุดข้อมูลนี้: {out_count} จุด (จัดการเรียบร้อยแล้ว)")

    # 4. กราฟเปรียบเทียบข้อมูลก่อนพยากรณ์
    fig_clean = go.Figure()
    fig_clean.add_trace(go.Scatter(x=df[time_col], y=df[target_col], name="ข้อมูลดิบ", line=dict(color='silver')))
    fig_clean.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean[target_col], name="ข้อมูลที่ปรับแต่งแล้ว", line=dict(color='#00CC96')))
    
    # ไฮไลต์หน้าต่างข้อมูลล่าสุด (Lag) ที่จะถูกป้อนเข้า AI
    last_idx = df_clean[time_col].iloc[-lag:]
    last_val = df_clean[target_col].iloc[-lag:]
    fig_clean.add_trace(go.Scatter(x=last_idx, y=last_val, name="หน้าต่างข้อมูลที่ AI ใช้ทาย", line=dict(color='red', width=4)))
    
    fig_clean.update_layout(template="plotly_white", title="การเตรียมข้อมูลฐานสำหรับการพยากรณ์")
    st.plotly_chart(fig_clean, use_container_width=True)

    # 5. ตั้งค่าการพยากรณ์
    st.divider()
    horizon = st.number_input(
    "🚀 จำนวนก้าวที่ต้องการพยากรณ์ไปข้างหน้า",
    min_value=1,
    max_value=10000,
    value=24
)
    if st.button("✨ เริ่มพยากรณ์อนาคต"):
        with st.spinner("🤖 AI กำลังประมวลผล..."):
            series = df_clean[target_col].values
            if len(series) < lag:
                st.error("ข้อมูลมีน้อยกว่าค่า Lag ที่ตั้งไว้")
                st.stop()
            last_window = series[-lag:]
            model = artifact["model"]

            future = model.forecast(
            last_window,
            steps=horizon
        )

            future = np.array(future).flatten()

            # smooth เล็กน้อยเพื่อให้กราฟดูสวย
           # future = pd.Series(future).rolling(window=20).mean().bfill().values

            # 6. แสดงผลลัพธ์
            st.subheader("📈 ผลลัพธ์การพยากรณ์")
            
            # คำนวณแกน X สำหรับอนาคต
            last_time = df_clean[time_col].iloc[-1]
            future_x = np.arange(last_time + 1, last_time + 1 + len(future))

            fig_res = go.Figure()
            # เส้นประวัติ (เฉพาะส่วนท้ายเพื่อให้เห็นชัด)
            fig_res.add_trace(go.Scatter(x=df_clean[time_col].iloc[-lag*2:], y=df_clean[target_col].iloc[-lag*2:], name="ข้อมูลล่าสุด"))
            # เส้นพยากรณ์
            fig_res.add_trace(go.Scatter(x=future_x, y=future, name="ค่าพยากรณ์", line=dict(color='red', width=3)))
            
            fig_res.update_layout(template="plotly_white", title=f"พยากรณ์ล่วงหน้า {horizon} ช่วงเวลา")
            st.plotly_chart(fig_res, use_container_width=True)

            if model_type == "lstm":
                st.write("is_fitted:", model.is_fitted)
                print("device:", next(model.model.parameters()).device)
                print("scaler min:", model.scaler.data_min_)
                print("scaler max:", model.scaler.data_max_)
            
            # 7. ส่งออกข้อมูล
            result_df = pd.DataFrame({
                "Time_Index": future_x,
                "Forecast_Value": future
            })
            
            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                result_df["Time_Index"] = result_df["Time_Index"].astype(int)
                result_df["Forecast_Value"] = result_df["Forecast_Value"].astype(float)
                st.dataframe(result_df, use_container_width=True)
            with col_d2:
                st.write("💾 ดาวน์โหลดไฟล์")
                st.download_button("📥 CSV", result_df.to_csv(index=False).encode("utf-8-sig"), "forecast.csv")
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    result_df.to_excel(writer, index=False)
                st.download_button("📥 Excel", output.getvalue(), "forecast.xlsx")
