from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
st.set_page_config(page_title='PowerGrid Cyber AI',page_icon='⚡',layout='wide')
st.title('⚡ PowerGrid Cyber AI');st.caption('لوحة محاكاة آمنة — لا اتصال بشبكة حقيقية')
@st.cache_data
def data():
 p=R/'predictions_test.csv';return pd.read_csv(p) if p.exists() else pd.DataFrame()
d=data()
with st.sidebar:
 page=st.radio('الصفحة',['Overview','Live Risk','Evaluation','Feature Importance','About']);st.info('بيانات تاريخية/تجريبية فقط')
if page=='Overview':
 st.subheader('ملخص');
 if d.empty:st.warning('شغل التدريب والتقييم أولاً')
 else:
  e=int((d.y_true!=d.rf_prediction).sum());a,b,c=st.columns(3);a.metric('حالات الاختبار',len(d));b.metric('أخطاء RF',e);c.metric('Accuracy',f'{1-e/len(d):.2%}');st.bar_chart(d.y_true.value_counts());st.dataframe(d.tail(30),use_container_width=True)
elif page=='Live Risk':

    st.subheader('محاكاة Risk Score')
    st.caption('يجب أن يكون مجموع الاحتمالات 100% بالضبط.')

    normal = st.slider(
        'P(Normal)',
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01
    )

    remaining = max(0.0, 1.0 - normal)

    if remaining <= 0.0:
        fault = 0.0
        attack = 0.0
    else:
        fault = st.slider(
            'P(Fault)',
            min_value=0.0,
            max_value=remaining,
            value=min(0.25, remaining),
            step=0.01
        )
        attack = remaining - fault

    probabilities = {
        'Normal': normal,
        'Fault': fault,
        'Attack': attack
    }

    total = sum(probabilities.values())
    probabilities['Attack'] += 1.0 - total
    probabilities['Attack'] = max(0.0, probabilities['Attack'])

    predicted = max(probabilities, key=probabilities.get)
    risk_score = 1.0 - probabilities['Normal']

    c1, c2, c3 = st.columns(3)
    c1.metric('الفئة', predicted)
    c2.metric('Risk Score', f'{risk_score:.2%}')
    c3.metric('P(Attack)', f'{probabilities["Attack"]:.2%}')

    st.dataframe(
        pd.DataFrame({
            'Class': list(probabilities.keys()),
            'Probability': [
                f'{value:.2%}'
                for value in probabilities.values()
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(pd.Series(probabilities))

    if normal >= 0.99:
        st.success('الحالة طبيعية تقريباً: Risk Score منخفض جداً.')
    elif risk_score >= 0.80:
        st.error('تنبيه محاكاة: الحالة غير طبيعية بدرجة مرتفعة.')
    elif risk_score >= 0.50:
        st.warning('تنبيه محاكاة متوسط: تحتاج الحالة إلى مراجعة.')
    else:
        st.info('الخطر المحاكى منخفض.')
elif page=='Evaluation':
 st.subheader('التقييم');
 if d.empty:st.warning('شغل التدريب والتقييم أولاً')
 else:
  model=st.selectbox('النموذج',['random_forest','xgboost']);col='rf_prediction' if model=='random_forest' else 'xgb_prediction';m=pd.crosstab(d.y_true,d[col]).reindex(index=['Normal','Fault','Attack'],columns=['Normal','Fault','Attack'],fill_value=0);st.dataframe(m,use_container_width=True);a,b=st.columns(2);a.metric('Attack → Fault',int(((d.y_true=='Attack')&(d[col]=='Fault')).sum()));b.metric('Fault → Attack',int(((d.y_true=='Fault')&(d[col]=='Attack')).sum()))
elif page=='Feature Importance':
 p=R/'top_20_features.csv';st.subheader('أهم 20 خاصية');
 if p.exists():
  x=pd.read_csv(p);st.dataframe(x,use_container_width=True);st.bar_chart(x.set_index('feature')['importance'])
 else:st.info('شغل src/explainability.py')
else:st.subheader('عن النظام');st.write('نظام أكاديمي لتصنيف Normal وFault وAttack. Risk Score مؤشر دعم قرار وليس أمراً تشغيلياً.')
