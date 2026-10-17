import Foundation

/// Minimal runtime language switch: zh (default) / en, stored in appLang.
enum L10n {
    static var lang: String {
        UserDefaults.standard.string(forKey: "appLang") ?? "zh"
    }

    /// Usage: L10n.t("中文", "English")
    static func t(_ zh: String, _ en: String) -> String {
        lang == "en" ? en : zh
    }
}

extension L10n {
    static func semanticName(_ name: String) -> String {
        let translations = [
            "wall": "墙面", "floor": "地面", "ceiling": "天花板", "chair": "椅子",
            "table": "桌子", "desk": "书桌", "cabinet": "柜子", "bookshelf": "书架",
            "bookcase": "书架", "shelf": "置物架", "sofa": "沙发", "couch": "沙发",
            "bed": "床", "door": "门", "window": "窗户", "curtain": "窗帘",
            "picture": "挂画", "sink": "水槽", "toilet": "马桶", "bathtub": "浴缸",
            "refrigerator": "冰箱", "refridgerator": "冰箱", "counter": "台面",
            "shower curtain": "浴帘", "showercurtain": "浴帘", "otherfurniture": "其他家具",
            "other furniture": "其他家具", "blackboard": "黑板", "whiteboard": "白板",
            "monitor": "显示器", "tv": "电视", "trash can": "垃圾桶", "trashcan": "垃圾桶",
            "office chair": "办公椅", "stool": "凳子", "lamp": "灯", "unknown": "未分类",
            "printer": "打印机", "projector": "投影仪", "robot": "机器人", "toolbox": "工具箱",
            "trash bin": "垃圾桶", "fan": "风扇", "computer": "电脑", "box": "箱子", "tripod": "三脚架",
            "table-like object (desk/table unresolved)": "桌类物体", "curtain-like object (subtype unresolved)": "帘类物体"
        ]
        return t(translations[name.lowercased()] ?? name, name)
    }
}
