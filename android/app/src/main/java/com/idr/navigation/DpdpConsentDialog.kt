package com.idr.navigation

import android.app.AlertDialog
import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.widget.CheckBox
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView

/**
 * DPDP Act 2023 Multilingual Consent Dialog.
 *
 * Displays an explicit, unbundled notice and consent request in compliance with
 * the Digital Personal Data Protection Act 2023 (India).
 *
 * Supports Hindi and English language toggling.
 */
class DpdpConsentDialog(
    private val context: Context,
    private val consentManager: DpdpConsentManager,
    private val onConsentResolved: (granted: Boolean) -> Unit
) {

    private enum class Language {
        ENGLISH, HINDI
    }

    private var currentLanguage = Language.ENGLISH

    fun show() {
        val dialogView = createConsentView()

        val dialog = AlertDialog.Builder(context)
            .setTitle(getTitle())
            .setView(dialogView)
            .setCancelable(false)
            .setPositiveButton(getAcceptText(), null)
            .setNegativeButton(getDeclineText()) { _, _ ->
                consentManager.withdrawConsent()
                onConsentResolved(false)
            }
            .create()

        dialog.setOnShowListener {
            val positiveBtn = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
            positiveBtn.setOnClickListener {
                val cbLogging = dialogView.findViewById<CheckBox>(R_ID_CB_LOGGING)
                val cbCloud = dialogView.findViewById<CheckBox>(R_ID_CB_CLOUD)

                consentManager.grantConsent(
                    sensorLogging = cbLogging?.isChecked ?: false,
                    cloudUpload = cbCloud?.isChecked ?: false
                )
                dialog.dismiss()
                onConsentResolved(true)
            }
        }

        dialog.show()
    }

    private fun createConsentView(): View {
        val root = android.widget.LinearLayout(context).apply {
            orientation = android.widget.LinearLayout.VERTICAL
            setPadding(40, 20, 40, 20)
        }

        // Language toggle
        val langGroup = RadioGroup(context).apply {
            orientation = RadioGroup.HORIZONTAL
        }
        val rbEn = RadioButton(context).apply {
            text = "English"
            isChecked = (currentLanguage == Language.ENGLISH)
        }
        val rbHi = RadioButton(context).apply {
            text = "हिन्दी (Hindi)"
            isChecked = (currentLanguage == Language.HINDI)
        }
        langGroup.addView(rbEn)
        langGroup.addView(rbHi)
        root.addView(langGroup)

        // Notice Body
        val tvNotice = TextView(context).apply {
            id = R_ID_TV_NOTICE
            text = getNoticeText()
            textSize = 14f
            setPadding(0, 16, 0, 16)
        }
        root.addView(tvNotice)

        // Optional Opt-Ins
        val cbLogging = CheckBox(context).apply {
            id = R_ID_CB_LOGGING
            text = getLoggingText()
            isChecked = false
        }
        root.addView(cbLogging)

        val cbCloud = CheckBox(context).apply {
            id = R_ID_CB_CLOUD
            text = getCloudText()
            isChecked = false
        }
        root.addView(cbCloud)

        langGroup.setOnCheckedChangeListener { _, checkedId ->
            currentLanguage = if (checkedId == rbHi.id) Language.HINDI else Language.ENGLISH
            tvNotice.text = getNoticeText()
            cbLogging.text = getLoggingText()
            cbCloud.text = getCloudText()
        }

        return root
    }

    private fun getTitle(): String = when (currentLanguage) {
        Language.ENGLISH -> "Data Privacy & Consent (DPDP Act 2023)"
        Language.HINDI -> "डेटा गोपनीयता एवं सहमति (DPDP अधिनियम 2023)"
    }

    private fun getNoticeText(): String = when (currentLanguage) {
        Language.ENGLISH -> """
            Purpose of Processing:
            IDR processes device sensors (accelerometer, gyroscope, barometer) and GNSS signals locally on this device to provide accurate vehicle dead reckoning during GPS outages (flyovers, tunnels, dense urban areas).
            
            Data Protection Guarantees:
            • 100% On-Device Processing by default.
            • Zero mandatory cloud transmission.
            • You have the right to withdraw consent and erase stored data at any time under DPDP Act 2023 §6.
        """.trimIndent()
        Language.HINDI -> """
            डेटा प्रसंस्करण का उद्देश्य:
            IDR आपके डिवाइस के सेंसर (एक्सेलेरोमीटर, जायरोस्कोप, बैरोमीटर) और जीपीएस डेटा को इसी डिवाइस पर स्थानीय रूप से प्रोसेस करता है, ताकि जीपीएस सिग्नल न होने पर (फ्लाईओवर, सुरंग, घनी बस्तियां) भी सटीक नेविगेशन मिल सके।
            
            डेटा सुरक्षा की गारंटी:
            • डिफ़ॉल्ट रूप से 100% डेटा इसी डिवाइस पर प्रोसेस होता है।
            • क्लाउड पर कोई डेटा नहीं भेजा जाता।
            • DPDP अधिनियम 2023 की धारा 6 के तहत आप किसी भी समय अपनी सहमति वापस ले सकते हैं।
        """.trimIndent()
    }

    private fun getLoggingText(): String = when (currentLanguage) {
        Language.ENGLISH -> "Allow local sensor logging for motion calibration (Optional)"
        Language.HINDI -> "मोशन कैलिब्रेशन के लिए स्थानीय सेंसर डेटा संग्रह की अनुमति दें (वैकल्पिक)"
    }

    private fun getCloudText(): String = when (currentLanguage) {
        Language.ENGLISH -> "Share anonymized trajectory metrics to improve navigation models (Optional)"
        Language.HINDI -> "मॉडल सुधारने के लिए अनाम नेविगेशन आंकड़े साझा करें (वैकल्पिक)"
    }

    private fun getAcceptText(): String = when (currentLanguage) {
        Language.ENGLISH -> "Accept & Continue"
        Language.HINDI -> "स्वीकार करें"
    }

    private fun getDeclineText(): String = when (currentLanguage) {
        Language.ENGLISH -> "Decline"
        Language.HINDI -> "अस्वीकार करें"
    }

    companion object {
        private const val R_ID_TV_NOTICE = 1001
        private const val R_ID_CB_LOGGING = 1002
        private const val R_ID_CB_CLOUD = 1003
    }
}
