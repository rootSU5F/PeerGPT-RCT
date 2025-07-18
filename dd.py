import streamlit as st

# عنوان الصفحة
st.title("🎉 تجربتي الأولى مع Streamlit")

# حقل إدخال
user_input = st.text_input("اكتب أي شيء:")

# زر
if st.button("اضغط هنا"):
    st.success(f"أهلًا بك! أنت كتبت: {user_input}")
