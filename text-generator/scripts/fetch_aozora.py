"""
青空文庫からテキストを取得してひらがなに変換
"""

import re
import urllib.request
import pykakasi


def fetch_text(url: str) -> str:
    """URLからテキストを取得"""
    with urllib.request.urlopen(url) as response:
        # 青空文庫はShift_JISが多い
        return response.read().decode("shift_jis", errors="ignore")


def clean_aozora(text: str) -> str:
    """青空文庫の注記を削除"""
    # ルビを使う: 漢字《かんじ》 → かんじ（ルビの方を採用）
    text = re.sub(r"[一-龯々]+《([^》]+)》", r"\1", text)
    # 残ったルビ記号を削除
    text = re.sub(r"《[^》]+》", "", text)
    # 注記を削除: ［＃...］
    text = re.sub(r"［＃[^］]+］", "", text)
    # 丸括弧の注記も削除: （注記）
    text = re.sub(r"（[^）]*）", "", text)
    # ヘッダー・フッター削除（---で区切られている）
    if "---" in text:
        parts = text.split("---")
        if len(parts) >= 3:
            text = parts[1]
    return text.strip()


def to_hiragana(text: str) -> str:
    """漢字・カタカナをひらがなに変換"""
    kakasi = pykakasi.kakasi()
    kakasi.setMode("K", "H")  # カタカナ → ひらがな
    kakasi.setMode("J", "H")  # 漢字 → ひらがな
    converter = kakasi.getConverter()
    return converter.do(text)


def main():
    # 宮沢賢治「銀河鉄道の夜」
    url = "https://www.aozora.gr.jp/cards/000081/files/456_15050.html"

    print("テキスト取得中...")
    try:
        html = fetch_text(url)
    except Exception as e:
        print(f"取得失敗: {e}")
        print("ローカルサンプルを作成します...")
        # 代わりに桃太郎のサンプルを使用
        sample = """
むかしむかし、あるところにおじいさんとおばあさんがいました。
おじいさんはやまへしばかりに、おばあさんはかわへせんたくにいきました。
おばあさんがかわでせんたくをしていると、おおきなももがどんぶらこどんぶらことながれてきました。
おばあさんはそのももをひろいあげて、いえにもってかえりました。
おじいさんがかえってきて、ももをきってみると、なかからおとこのこがうまれました。
ふたりはそのこをももたろうとなづけました。
ももたろうはすくすくとそだち、おおきくなりました。
あるひ、ももたろうはおにがしまへおにたいじにいくことにしました。
おばあさんにきびだんごをつくってもらい、たびにでました。
みちのとちゅうで、いぬにあいました。
いぬはきびだんごをもらって、ももたろうのけらいになりました。
つぎに、さるにあいました。
さるもきびだんごをもらって、けらいになりました。
さいごに、きじにあいました。
きじもきびだんごをもらって、けらいになりました。
ももたろうは、いぬ、さる、きじをつれて、おにがしまにつきました。
おにたちとたたかい、みごとにかちました。
おにたちからたからものをもらい、むらにかえりました。
おじいさんとおばあさんはとてもよろこびました。
めでたしめでたし。
"""
        # 繰り返して量を増やす
        hiragana = (sample.strip() + "\n") * 100
        output_path = "data/hiragana.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(hiragana)
        print(f"保存完了: {output_path}")
        print(f"文字数: {len(hiragana)}")
        return

    # HTMLからbody部分を抽出
    match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        text = html

    # HTMLタグ削除
    text = re.sub(r"<[^>]+>", "", text)
    text = clean_aozora(text)

    print("ひらがな変換中...")
    hiragana = to_hiragana(text)

    # 余分な空白を整理
    hiragana = re.sub(r"\s+", "\n", hiragana)

    # 保存
    output_path = "data/hiragana.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(hiragana)

    print(f"保存完了: {output_path}")
    print(f"文字数: {len(hiragana)}")


if __name__ == "__main__":
    main()
