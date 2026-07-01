import streamlit as st
import struct

# --- 核心转换逻辑函数 ---
def convert_value_to_hex(value_str, data_type, endianness, encoding='utf-8'):
    """将输入的字符串和类型转换为 Hex 字节串"""
    
    # 确定字节序符号：< (小端/Little-Endian) 或 > (大端/Big-Endian)
    endian_symbol = '<' if endianness == 'Little-Endian' else '>'
    
    try:
        if data_type in ['byte', 'ubyte', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']:
            # 整数类型处理
            value = int(value_str)
            # 确定 struct 格式代码
            format_code = {
                'byte': 'b', 'ubyte': 'B',
                'short': 'h', 'ushort': 'H',
                'int': 'i', 'uint': 'I',
                'long': 'l', 'ulong': 'L'
            }[data_type]
            
            # 打包数据并转换为 Hex
            byte_data = struct.pack(endian_symbol + format_code, value)
            
        elif data_type in ['float', 'double']:
            # 浮点数类型处理 (IEEE 754)
            value = float(value_str)
            format_code = {'float': 'f', 'double': 'd'}[data_type]
            byte_data = struct.pack(endian_symbol + format_code, value)
            
        elif data_type in ['ascii', 'utf-8', 'unicode']:
            # 字符串类型处理
            byte_data = value_str.encode(encoding)
            
        else:
            return "不支持的数据类型选择"
            
        # 将字节数组格式化为带空格的 Hex 字符串
        return byte_data.hex(' ').upper()
    
    except ValueError:
        return "错误：输入值与所选类型不匹配或格式错误。"
    except Exception as e:
        return f"发生未知错误: {e}"

# --- Streamlit 界面构建 ---

st.set_page_config(page_title="工业数据 Hex 转换工具", layout="centered")

st.title("⚙️ 数值类型转换工具")
st.markdown("将十进制、浮点数或字符串转换为 Hex 字节串，常用于工业通信 (如 Modbus)。")

# --- 输入区 ---
value_str = st.text_input("请输入等待转换的值:", value="12345")

# --- 类型选择区 (模仿您的截图) ---
st.subheader("类型数据:")

# 整数
col_int_1, col_int_2, col_int_3, col_int_4, col_int_5, col_int_6, col_int_7, col_int_8 = st.columns(8)
integer_types = ['byte', 'short', 'ushort', 'int', 'uint', 'long', 'ulong']
selected_type = None
# 使用 Session State 来确保每次只有一个按钮被选中
if 'selected_type' not in st.session_state:
    st.session_state.selected_type = None

# Streamlit 的 Radio 按钮更适合单选，但为了模拟截图样式，我们使用按钮
type_map = {
    col_int_1: 'byte', col_int_2: 'short', col_int_3: 'ushort', col_int_4: 'int',
    col_int_5: 'uint', col_int_6: 'long', col_int_7: 'ulong'
}

# 动态生成按钮并管理状态
with st.container():
    st.markdown("###### 整数数据")
    cols = st.columns(len(integer_types))
    for col, dtype in zip(cols, integer_types):
        if col.button(dtype, key=f"btn_{dtype}", use_container_width=True):
            st.session_state.selected_type = dtype

# 浮点数
with st.container():
    st.markdown("###### 浮点数据")
    col_float_1, col_float_2, _ = st.columns([1, 1, 6])
    float_types = ['float', 'double']
    for col, dtype in zip([col_float_1, col_float_2], float_types):
        if col.button(dtype, key=f"btn_{dtype}", use_container_width=True):
            st.session_state.selected_type = dtype

# 字符数据
with st.container():
    st.markdown("###### 字符数据")
    col_str_1, col_str_2, col_str_3, col_str_4, col_str_5 = st.columns(5)
    string_types = ['ascii', 'ansi', 'utf-8', 'utf-16', 'utf-32']
    for col, dtype in zip([col_str_1, col_str_2, col_str_3, col_str_4, col_str_5], string_types):
        if col.button(dtype, key=f"btn_{dtype}", use_container_width=True):
            st.session_state.selected_type = dtype
            
# --- 额外配置 ---
st.markdown("---")
if st.session_state.selected_type not in ['ascii', 'utf-8', 'unicode', 'utf-16', 'utf-32']:
    endianness = st.radio(
        "选择字节序 (Endianness):",
        ('Little-Endian', 'Big-Endian'),
        horizontal=True
    )
else:
    # 字符串不需要字节序，但需要编码
    endianness = 'N/A' # 占位
    
# --- 转换和输出 ---
st.markdown("---")
if st.button("转换", type="primary"):
    if st.session_state.selected_type is None:
        st.warning("请先选择一种数据类型!")
    else:
        # 针对字符串类型，编码即为 'data_type'
        encoding = st.session_state.selected_type if st.session_state.selected_type in ['ascii', 'utf-8', 'unicode', 'utf-16', 'utf-32'] else 'utf-8'
        
        hex_result = convert_value_to_hex(
            value_str=value_str,
            data_type=st.session_state.selected_type,
            endianness=endianness,
            encoding=encoding
        )
        
        st.subheader("输出 (Hex)")
        st.code(hex_result)
        
st.markdown("---")
st.caption(f"当前选中的类型: **{st.session_state.selected_type or '无'}**")