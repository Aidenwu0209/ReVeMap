import SwiftUI

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

/// Shared language control; preserves the current navigation and appLang binding.
struct LanguageToggle: View {
    @AppStorage("appLang") private var lang = "zh"

    var body: some View {
        Group {
            if #available(iOS 26, *) {
                segments.glassEffect(.regular, in: .capsule)
            } else {
                segments.background(.ultraThinMaterial, in: Capsule())
            }
        }
        .fixedSize()
        .accessibilityElement(children: .contain)
        .accessibilityLabel(L10n.t("切换语言", "Switch language"))
    }

    private var segments: some View {
        HStack(spacing: 0) {
            languageButton("中", value: "zh", accessibilityName: "中文")
            languageButton("EN", value: "en", accessibilityName: "English")
        }
        .padding(4)
        .animation(.easeInOut(duration: 0.16), value: lang)
    }

    private func languageButton(_ title: String, value: String, accessibilityName: String) -> some View {
        Button {
            lang = value
        } label: {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(lang == value ? Color.primary : Color.secondary)
                .frame(width: 44, height: 24)
                .background {
                    if lang == value {
                        Capsule().fill(Color.primary.opacity(0.16))
                            .overlay {
                                Capsule().strokeBorder(Color.primary.opacity(0.08), lineWidth: 0.5)
                            }
                    }
                }
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityName)
        .accessibilityAddTraits(lang == value ? .isSelected : [])
        .accessibilityIdentifier("language-" + value)
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
