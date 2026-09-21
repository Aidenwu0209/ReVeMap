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
