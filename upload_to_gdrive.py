import os
import mimetypes
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ==================== 配置区域 ====================
# 1. 本地数据集的根路径
LOCAL_DATASETS_DIR = ".datasets/intraoral"

# 2. 凭证文件 credentials.json 所在的绝对路径目录
CREDENTIALS_DIR = "/home/zychen/Documents/intraoral_code"

# 3. Google Drive 上已有的 【App Photo】 文件夹的 ID
# 请去网页端进入 App Photo 文件夹，复制地址栏最后一串 ID 填在这里
GDRIVE_PARENT_FOLDER_ID = "1J8Ko4wIwJBVzJNFKQuNvSf0JT_kfKFXk"

# 4. 指定需要处理的人名列表
TARGET_NAMES = ["amy", "chenghao", "zeyu", "garris"] 

# Google Drive API 权限范围
SCOPES = ['https://www.googleapis.com/auth/drive.file']
# =================================================

def get_gdrive_service():
    """获取并认证 Google Drive 服务"""
    creds = None
    token_path = os.path.join(CREDENTIALS_DIR, 'token.json')
    credentials_path = os.path.join(CREDENTIALS_DIR, 'credentials.json')
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"未在指定路径找到凭证文件: {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            
            # --- 修改这里：改用控制台输入模式 ---
            # 它不会尝试打开 localhost，而是让你复制 URL 到浏览器，然后把回传的 code 贴回终端
            creds = flow.run_console() 
            # ----------------------------------
        
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)

def find_or_create_folder(service, folder_name, parent_id):
    """检查父目录下是否存在同名文件夹，存在则返回其 ID，不存在则新建"""
    # 查询语法：名字为 folder_name，父目录为 parent_id，且类型为文件夹，且未被删除
    query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    try:
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        
        if items:
            print(f"检测到云端已存在文件夹: 【{folder_name}】，直接复用。ID: {items[0]['id']}")
            return items[0]['id']
        else:
            # 不存在则新建
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            print(f"成功在云端新建文件夹: 【{folder_name}】，ID: {folder['id']}")
            return folder['id']
    except Exception as e:
        print(f"查找或创建文件夹 【{folder_name}】 失败: {e}")
        return None

def upload_file_to_gdrive(service, local_file_path, filename, parent_folder_id):
    """上传单个文件到 Google Drive 的指定文件夹下"""
    mime_type, _ = mimetypes.guess_type(local_file_path)
    if not mime_type:
        mime_type = 'application/octet-stream'

    file_metadata = {
        'name': filename,
        'parents': [parent_folder_id]
    }
    
    media = MediaFileUpload(local_file_path, mimetype=mime_type, resumable=True)
    
    try:
        print(f"  正在上传: {filename} ...")
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"  --> 上传成功！云端文件 ID: {file.get('id')}")
    except Exception as e:
        print(f"  --> 上传失败 【{filename}】: {e}")

def main():
    # 1. 初始化 Drive 服务
    service = get_gdrive_service()
    
    if not os.path.exists(LOCAL_DATASETS_DIR):
        print(f"错误: 本地路径不存在 {LOCAL_DATASETS_DIR}")
        return

    # 2. 先在 App Photo 文件夹下面创建（或定位）总的 "process" 文件夹
    print("正在检查/创建云端总目录 'process'...")
    process_folder_id = find_or_create_folder(service, "process", GDRIVE_PARENT_FOLDER_ID)
    if not process_folder_id:
        print("核心错误: 无法获取或创建总 process 文件夹，程序终止。")
        return

    # 3. 遍历本地目录，处理每个人名
    for name in TARGET_NAMES:
        name_path = os.path.join(LOCAL_DATASETS_DIR, name)
        
        if not os.path.exists(name_path) or not os.path.isdir(name_path):
            print(f"\n跳过: 本地未找到人名目录 【{name}】")
            continue
        
        # 找出该人名目录下所有以 'process.zip' 结尾的文件
        zip_files = [f for f in os.listdir(name_path) if f.endswith('process.zip') and os.path.isfile(os.path.join(name_path, f))]
        
        if not zip_files:
            print(f"\n提示: 本地 【{name}】 目录下没有检测到 *process.zip 文件。")
            continue
            
        print(f"\n开始处理成员: 【{name}】")
        
        # 4. 在云端 "process" 文件夹内部，为该成员新建（或复用）人名文件夹
        name_folder_id = find_or_create_folder(service, name, process_folder_id)
        
        if not name_folder_id:
            print(f"错误: 无法为 【{name}】 创建云端专属文件夹，跳过此人。")
            continue
            
        # 5. 批量上传 zip 文件到该人名文件夹中
        for zip_file in zip_files:
            full_local_path = os.path.join(name_path, zip_file)
            upload_file_to_gdrive(service, full_local_path, zip_file, name_folder_id)

if __name__ == '__main__':
    main()