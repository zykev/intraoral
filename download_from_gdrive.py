import io
import os
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ==================== 配置区域 ====================
# 1. 云端要下载的文件夹 ID
#    从 https://drive.google.com/drive/u/1/folders/10TZzBgRp6LjVlYIgbyj8Hq_-DqsCHhIl 取最后一段
DRIVE_FOLDER_ID = "10TZzBgRp6LjVlYIgbyj8Hq_-DqsCHhIl"

# 2. 本地保存的根目录（会在此目录下重建云端的文件夹结构）
#    注意是 intraoral_adjust/sample —— 每个 sample id 一个文件夹，
#    里面放该 sample 的 {D,F,L,R,U}.{png,txt,json}。
#    用仓库内的 .datasets/ 软链，避免写死存储卷路径。
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DOWNLOAD_DIR = os.path.join(
    REPO_ROOT, ".datasets", "intraoral_adjust", "sample"
)

# 3. 凭证文件 credentials.json / token.json 所在的绝对路径目录
CREDENTIALS_DIR = os.path.dirname(REPO_ROOT)

# 4. 是否跳过本地已存在的同名文件（用于断点续传）
SKIP_EXISTING = True

# Google Drive API 权限范围
#
# 注意：这里必须用完整的 drive 权限，不能用 drive.file。原因有二：
#   1) drive.file 只能访问“本应用自己创建/打开过”的文件，而 DRIVE_FOLDER_ID
#      指向的文件夹是别的应用/人创建的，drive.file 下会看不到它；
#   2) 现有 token.json 当初就是以 drive 授权生成的，刷新时 google-auth 会把这里的
#      scopes 发给 Google，若与 token 不一致会直接报 `invalid_scope: Bad Request`。
SCOPES = ['https://www.googleapis.com/auth/drive']
# =================================================

# Google 原生文档类型 → 导出格式（这些格式没有直接二进制可下载）
GOOGLE_EXPORT_MAP = {
    'application/vnd.google-apps.document': 'application/pdf',
    'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.google-apps.presentation': 'application/pdf',
    'application/vnd.google-apps.drawing': 'image/png',
}
GOOGLE_EXPORT_EXT = {
    'application/vnd.google-apps.document': '.pdf',
    'application/vnd.google-apps.spreadsheet': '.xlsx',
    'application/vnd.google-apps.presentation': '.pdf',
    'application/vnd.google-apps.drawing': '.png',
}


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


def list_children(service, parent_id):
    """列出某文件夹下的所有直接子项（自动翻页），返回 files 列表"""
    children = []
    page_token = None
    query = f"'{parent_id}' in parents and trashed = false"
    while True:
        try:
            results = service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageToken=page_token,
                pageSize=1000,
            ).execute()
        except Exception as e:
            print(f"  列出文件夹 [{parent_id}] 内容失败: {e}")
            return children

        children.extend(results.get('files', []))
        page_token = results.get('nextPageToken')
        if not page_token:
            break
    return children


def safe_local_path(dest_dir, name):
    """把云端文件名转成本地安全文件名，避免路径穿越/非法字符"""
    name = (name or "").replace("/", "_").replace("\\", "_").strip()
    if name in ("", ".", ".."):
        name = "_unnamed_"
    return os.path.join(dest_dir, name)


def download_file(service, file_meta, dest_path, skip_existing=True):
    """下载单个普通文件到本地（流式写盘，适配大文件）"""
    file_id = file_meta['id']
    name = file_meta['name']

    if skip_existing and os.path.isfile(dest_path):
        # 若能拿到云端大小，则比对大小；否则仅凭存在即跳过
        remote_size = file_meta.get('size')
        if remote_size is None or os.path.getsize(dest_path) == int(remote_size):
            print(f"  跳过（本地已存在）: {name}")
            return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # 先写临时文件，下载完成再改名，避免中断留下半个文件
    tmp_path = dest_path + ".part"

    try:
        print(f"  正在下载: {name} ...")
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(tmp_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"    {int(status.progress() * 100):3d}%", end="\r")
        os.replace(tmp_path, dest_path)
        print(f"  --> 下载成功: {os.path.basename(dest_path)}          ")
    except Exception as e:
        print(f"  --> 下载失败 【{name}】: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def export_google_doc(service, file_meta, dest_dir):
    """导出 Google 原生文档（Docs/Sheets/Slides/Drawing）为本地文件"""
    mime = file_meta['mimeType']
    name = file_meta['name']
    export_mime = GOOGLE_EXPORT_MAP.get(mime)
    if not export_mime:
        print(f"  跳过（不支持的 Google 文档类型 {mime}）: {name}")
        return

    ext = GOOGLE_EXPORT_EXT.get(mime, "")
    dest_path = safe_local_path(dest_dir, name)
    if not dest_path.lower().endswith(ext):
        dest_path += ext

    if SKIP_EXISTING and os.path.isfile(dest_path):
        print(f"  跳过（本地已存在）: {os.path.basename(dest_path)}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    try:
        print(f"  正在导出: {name} -> {os.path.basename(dest_path)} ...")
        request = service.files().export_media(fileId=file_meta['id'], mimeType=export_mime)
        with io.FileIO(dest_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print(f"  --> 导出成功: {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"  --> 导出失败 【{name}】: {e}")


def download_folder_recursive(service, folder_id, local_dir, skip_existing=True):
    """递归下载云端文件夹及其所有子项到本地目录"""
    os.makedirs(local_dir, exist_ok=True)
    children = list_children(service, folder_id)

    for child in children:
        mime = child['mimeType']
        name = child['name']

        if mime == 'application/vnd.google-apps.folder':
            print(f"[目录] {name}")
            download_folder_recursive(
                service, child['id'], safe_local_path(local_dir, name), skip_existing
            )
        elif mime in GOOGLE_EXPORT_MAP:
            export_google_doc(service, child, local_dir)
        elif mime.startswith('application/vnd.google-apps.'):
            print(f"  跳过（无法下载的 Google 类型 {mime}）: {name}")
        else:
            download_file(service, child, safe_local_path(local_dir, name), skip_existing)


def main():
    service = get_gdrive_service()

    # 先确认目标文件夹可访问并拿到名字（可选，仅用于日志）
    try:
        meta = service.files().get(
            fileId=DRIVE_FOLDER_ID, fields="id, name, mimeType"
        ).execute()
        print(f"开始下载云端文件夹: 【{meta.get('name')}】 (ID: {DRIVE_FOLDER_ID})")
    except Exception as e:
        print(f"无法访问云端文件夹 {DRIVE_FOLDER_ID}: {e}")
        return

    print(f"保存到本地目录: {os.path.abspath(LOCAL_DOWNLOAD_DIR)}\n")
    download_folder_recursive(service, DRIVE_FOLDER_ID, LOCAL_DOWNLOAD_DIR, SKIP_EXISTING)
    print("\n全部下载完成。")


if __name__ == '__main__':
    main()
