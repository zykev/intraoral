import os
import mimetypes
from google.auth.exceptions import RefreshError
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
TARGET_NAMES = ["amy", "chenghao", "garris", "haozhou", "henry", "yinghao", "zeyu", "zeyu_new"]

# 5. 云端新建的目标文件夹名称（本次上传归档目录）
UPLOAD_FOLDER_NAME = "process0929"

# Google Drive API 权限范围
#
# 注意：这里必须用完整的 drive 权限，不能用 drive.file。原因有二：
#   1) drive.file 只能访问“本应用自己创建/打开过”的文件，而 GDRIVE_PARENT_FOLDER_ID
#      指向的 【App Photo】 文件夹是别的应用/人创建的，drive.file 下会看不到它；
#   2) 现有 token.json 当初就是以 drive 授权生成的，刷新时 google-auth 会把这里的
#      scopes 发给 Google，若与 token 不一致会直接报 `invalid_scope: Bad Request`。
SCOPES = ['https://www.googleapis.com/auth/drive']
# =================================================

def _run_manual_flow(flow):
    """无浏览器 / SSH 环境下的手动授权流程。

    新版 google-auth-oauthlib 已移除 run_console()，这里改成：
      1) 打印授权 URL，让你在本地浏览器打开并同意；
      2) 授权后浏览器会跳到打不开的 http://localhost（正常现象），
         把地址栏里完整 URL 复制回来（或直接贴 code）；
      3) 用 code 换取 token。
    """
    from urllib.parse import urlparse, parse_qs

    flow.redirect_uri = "http://localhost"  # 与 credentials.json 中注册的 redirect_uris 一致
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")

    print("\n================ 需要 Google 授权 ================")
    print("请用有权限的 Google 账号在本地浏览器打开下面的链接并点“允许”：\n")
    print(auth_url)
    print("\n授权后会跳到一个无法访问的 localhost 页面，这是正常的。")
    print("请把地址栏里**完整的那串 URL** 复制粘贴回来（也可只贴 code= 后面的值）。")
    print("================================================\n")

    raw = input("粘贴 URL 或 code: ").strip()
    if raw.startswith("http"):
        code = (parse_qs(urlparse(raw).query).get("code") or [None])[0]
    else:
        code = raw or None
    if not code:
        raise RuntimeError("未能从输入中解析出授权 code，请重试。")

    flow.fetch_token(code=code)
    return flow.credentials


def get_gdrive_service():
    """获取并认证 Google Drive 服务"""
    creds = None
    token_path = os.path.join(CREDENTIALS_DIR, 'token.json')
    credentials_path = os.path.join(CREDENTIALS_DIR, 'credentials.json')
    
    if os.path.exists(token_path):
        # 不传 SCOPES：沿用 token.json 中记录的 scope 去刷新，
        # 避免“脚本里的 scope 与 token 不一致”触发 invalid_scope。
        creds = Credentials.from_authorized_user_file(token_path)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                # 刷新被拒（如 invalid_scope / invalid_grant / token 被撤销）→ 丢弃旧凭证重新授权
                print(f"刷新 token 失败({e})，需要重新授权，已删除旧 token.json。")
                try:
                    os.remove(token_path)
                except OSError:
                    pass
                creds = None

        if not creds:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"未在指定路径找到凭证文件: {credentials_path}")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)

            # 手动授权（不依赖 localhost 回调，适配远程 SSH 环境）
            creds = _run_manual_flow(flow)

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
    # 文件在上传前消失（被移动/删除/失效软链接）→ 跳过，不算失败
    if not os.path.isfile(local_file_path):
        print(f"  跳过（本地文件不存在）: {filename}")
        return

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

def _escape_query_value(value):
    """转义 Drive 查询语句中的单引号"""
    return value.replace("\\", "\\\\").replace("'", "\\'")

def file_exists_in_folder(service, filename, parent_folder_id):
    """检查云端某个文件夹下是否已存在同名文件（用于断点续传/跳过重复上传）"""
    safe_name = _escape_query_value(filename)
    query = (f"name = '{safe_name}' and '{parent_folder_id}' in parents "
             f"and mimeType != 'application/vnd.google-apps.folder' and trashed = false")
    try:
        results = service.files().list(q=query, fields="files(id, name)").execute()
        return len(results.get('files', [])) > 0
    except Exception as e:
        print(f"    检查云端是否已存在 【{filename}】 失败: {e}")
        return False

def upload_folder_to_gdrive(service, local_folder_path, parent_folder_id, skip_existing=True):
    """递归上传本地文件夹（含子文件夹）到 Google Drive 指定父文件夹下"""
    # 本地文件夹不存在（可能已被移动 / 删除 / 是失效的软链接）→ 直接跳过，不中断整个上传
    if not os.path.isdir(local_folder_path):
        print(f"  跳过（本地不存在或不是文件夹）: {local_folder_path}")
        return

    folder_name = os.path.basename(os.path.normpath(local_folder_path))

    # 先读取目录内容；万一读取过程中目录被删/权限问题，也优雅跳过而不是崩溃
    try:
        entries = sorted(os.listdir(local_folder_path))
    except OSError as e:
        print(f"  跳过（无法读取本地目录）: {local_folder_path} ({e})")
        return

    cloud_folder_id = find_or_create_folder(service, folder_name, parent_folder_id)
    if not cloud_folder_id:
        print(f"  错误: 无法创建云端文件夹 【{folder_name}】，跳过该文件夹。")
        return

    for entry in entries:
        full_local_path = os.path.join(local_folder_path, entry)

        # 条目可能在遍历期间消失（被移动/删除），先确认存在再处理
        if not os.path.exists(full_local_path):
            print(f"  跳过（本地已不存在）: {entry}")
            continue

        if os.path.isdir(full_local_path):
            upload_folder_to_gdrive(service, full_local_path, cloud_folder_id, skip_existing)
        elif os.path.isfile(full_local_path):
            if skip_existing and file_exists_in_folder(service, entry, cloud_folder_id):
                print(f"  跳过（云端已存在）: {entry}")
                continue
            upload_file_to_gdrive(service, full_local_path, entry, cloud_folder_id)

def main():
    # 1. 初始化 Drive 服务
    service = get_gdrive_service()
    
    if not os.path.exists(LOCAL_DATASETS_DIR):
        print(f"错误: 本地路径不存在 {LOCAL_DATASETS_DIR}")
        return

    # 2. 先在 App Photo 文件夹下面创建（或定位）本次的目标文件夹 process0929
    print(f"正在检查/创建云端总目录 '{UPLOAD_FOLDER_NAME}'...")
    target_folder_id = find_or_create_folder(service, UPLOAD_FOLDER_NAME, GDRIVE_PARENT_FOLDER_ID)
    if not target_folder_id:
        print(f"核心错误: 无法获取或创建总 {UPLOAD_FOLDER_NAME} 文件夹，程序终止。")
        return

    # 3. 遍历本地目录，处理每个人名
    for name in TARGET_NAMES:
        name_path = os.path.join(LOCAL_DATASETS_DIR, name)

        if not os.path.exists(name_path) or not os.path.isdir(name_path):
            print(f"\n跳过: 本地未找到人名目录 【{name}】")
            continue

        # 找出该人名目录下所有以 'process' 结尾的文件夹
        process_dirs = [
            d for d in os.listdir(name_path)
            if d.endswith('process') and os.path.isdir(os.path.join(name_path, d))
        ]

        if not process_dirs:
            print(f"\n提示: 本地 【{name}】 目录下没有检测到 *process 文件夹。")
            continue

        print(f"\n开始处理成员: 【{name}】，共检测到 {len(process_dirs)} 个 process 文件夹")

        # 4. 在云端目标文件夹内部，为该成员新建（或复用）人名文件夹
        name_folder_id = find_or_create_folder(service, name, target_folder_id)

        if not name_folder_id:
            print(f"错误: 无法为 【{name}】 创建云端专属文件夹，跳过此人。")
            continue

        # 5. 递归上传每个 process 文件夹到该人名文件夹中
        for process_dir in sorted(process_dirs):
            full_local_path = os.path.join(name_path, process_dir)
            print(f"  --> 上传文件夹: {process_dir}")
            upload_folder_to_gdrive(service, full_local_path, name_folder_id)

if __name__ == '__main__':
    main()