# 🧪 Panduan Testing Person C via Streamlit App

## 📍 URL App
Buka browser dan akses: **http://localhost:8501**

---

## 🎯 Test Scenario 1: NER - Order ID dan Date

### Langkah:
1. Di Streamlit app, ketik pesan berikut di text input:
   ```
   Pesanan ORDER123 saya belum sampai sudah 5 hari
   ```
2. Tekan Enter atau klik Send
3. Klik expand **"🔍 Detected Info"** di bawah pesan Anda

### Expected Result:
- **Intent**: complaint_delivery (confidence ~90%)
- **Entities**: 
  - `order_id`: ORDER123
  - `date`: 5 hari

✅ **PASS** jika Order ID dan Date ter-extract dengan benar

---

## 🎯 Test Scenario 2: NER - Product dan Amount

### Langkah:
1. Ketik pesan:
   ```
   iPhone 15 Pro Max yang saya pesan Rp 25.000.000 tidak sesuai
   ```
2. Send dan expand "Detected Info"

### Expected Result:
- **Entities**:
  - `product`: iPhone 15 Pro Max (atau minimal "iPhone")
  - `amount`: Rp 25.000.000

✅ **PASS** jika Product dan Amount ter-extract

---

## 🎯 Test Scenario 3: NER - Person Name

### Langkah:
1. Ketik:
   ```
   Halo, nama saya Budi. Mau komplain pesanan #789456 yang rusak
   ```
2. Send dan check Detected Info

### Expected Result:
- **Entities**:
  - `order_id`: #789456
  - (Person name mungkin tidak muncul di entities display, tapi ter-extract di backend)

✅ **PASS** jika Order ID ter-extract

---

## 🎯 Test Scenario 4: NER - Multiple Entities

### Langkah:
1. Ketik:
   ```
   ORDER-98765 Samsung Galaxy S24 Ultra sudah 3 hari tracking tidak update
   ```
2. Check Detected Info

### Expected Result:
- **Entities**:
  - `order_id`: ORDER-98765
  - `product`: Samsung Galaxy S24 Ultra (atau minimal "Samsung")
  - `date`: 3 hari

✅ **PASS** jika minimal 2 dari 3 entities ter-extract

---

## 🎯 Test Scenario 5: Conversation Summary

### Langkah:
1. **CLEAR conversation** dulu (klik "🗑️ Clear Conversation" di sidebar)
2. Kirim pesan pertama:
   ```
   Halo, pesanan ORDER123 saya belum sampai
   ```
3. Kirim pesan kedua:
   ```
   Sudah 5 hari, harusnya 3 hari sampai
   ```
4. Kirim pesan ketiga:
   ```
   Tolong segera dicek ya
   ```
5. Lihat **sidebar kanan** bagian "📝 Conversation Summary"

### Expected Result:
Summary harus muncul dan mention:
- "keterlambatan" atau "belum sampai"
- "ORDER123"
- "5 hari" atau durasi

✅ **PASS** jika summary muncul dan relevans

---

## 🎯 Test Scenario 6: Summary - Produk Rusak

### Langkah:
1. Clear conversation
2. Kirim:
   ```
   iPhone 15 Pro Max yang saya terima rusak layarnya
   ```
3. Kirim:
   ```
   Nomor order INV-20240120001
   ```
4. Kirim:
   ```
   Mau replacement saja
   ```
5. Check sidebar summary

### Expected Result:
Summary mention:
- "rusak" atau "cacat"
- "replacement" atau "refund"
- "INV-20240120001" (optional)

✅ **PASS** jika summary capture topik utama

---

## 🎯 Test Scenario 7: Summary - Tracking

### Langkah:
1. Clear conversation
2. Kirim beberapa pesan:
   ```
   Gimana cara tracking pesanan?
   ```
   ```
   Sudah cek tapi statusnya stuck di 'Processing'
   ```
   ```
   Nomor ordernya ORDER789456
   ```
3. Check summary

### Expected Result:
Summary mention "tracking" atau "status"

✅ **PASS** jika summary generated

---

## 🎯 Test Scenario 8: Entity Extraction - Edge Cases

### Test 8a - No Entities
```
Halo, ada yang bisa dibantu?
```
**Expected**: Entities kosong atau minimal (OK jika tidak ada)

### Test 8b - Multiple Order IDs
```
ORDER123 dan ORDER456 belum sampai
```
**Expected**: Minimal 1 order ID ter-extract

### Test 8c - Indonesian Product
```
Sepatu Nike Air Jordan saya salah ukuran
```
**Expected**: Product "sepatu Nike" atau "Nike" ter-extract

---

## 📊 Checklist Testing Person C

Copy checklist ini dan mark setelah test:

### NER Testing
- [ ] Extract Order ID (ORDER123, ORD-456, #123456)
- [ ] Extract Product Name (iPhone, Samsung, Laptop, dll)
- [ ] Extract Date (kemarin, 5 hari, dll)
- [ ] Extract Amount (Rp 500.000, dll)
- [ ] Handle multiple entities in one message
- [ ] Handle no entities (empty message)

### Summarization Testing
- [ ] Summary muncul setelah 2+ pesan
- [ ] Summary untuk conversation keterlambatan
- [ ] Summary untuk conversation produk rusak
- [ ] Summary untuk conversation refund
- [ ] Summary mention order ID jika disebutkan
- [ ] Summary update saat conversation bertambah

### Integration Testing
- [ ] NER terintegrasi di "Detected Info"
- [ ] Summary muncul di sidebar
- [ ] Tidak ada error/crash saat digunakan
- [ ] Performance acceptable (tidak terlalu lambat)

---

## 🐛 Troubleshooting

### Summary Tidak Muncul
- Pastikan minimal 2 pesan sudah dikirim
- Check sidebar kanan, scroll ke bawah
- Coba clear conversation dan mulai lagi

### Entities Tidak Ter-extract
- Coba pesan yang lebih jelas (gunakan scenario di atas)
- Order ID harus format: ORDER123, ORD-456, #123456, INV-001
- Product name harus common: iPhone, Samsung, Laptop, Sepatu

### App Lambat
- Normal saat pertama kali (loading model)
- Setelah model loaded, seharusnya lebih cepat
- Jika terlalu lambat, check terminal untuk error

---

## 📸 Screenshot Expected Results

### Detected Info Example:
```
🔍 Detected Info
Intent: complaint_delivery (92%)
Entities: {'order_id': 'ORDER123', 'date': '5 hari'}
```

### Summary Example:
```
📝 Conversation Summary
Percakapan dengan 6 pesan. Pelanggan menghubungi terkait 
keterlambatan pengiriman. Nomor pesanan yang disebutkan: ORDER123.
```

---

## ✅ Success Criteria

**Minimum untuk PASS:**
1. ✅ NER dapat extract minimal 70% entities yang jelas
2. ✅ Summary generated untuk conversation 2+ pesan
3. ✅ Tidak ada crash/error saat normal usage
4. ✅ Integration dengan app.py berjalan lancar

**Bonus (Nice to Have):**
- Indonesian product names ter-extract
- Person names ter-extract
- Summary quality tinggi (mirip BART model)

---

## 🎓 Tips Testing

1. **Test dengan data realistic** - Gunakan kalimat seperti customer service beneran
2. **Test edge cases** - Empty message, multiple entities, dll
3. **Check console/terminal** - Lihat ada error atau tidak
4. **Compare dengan expected** - Gunakan test cases di `data/test_data_person_c.json`

---

**Happy Testing! 🚀**

Kalau ada yang tidak sesuai expected, catat dan bisa discuss untuk improvement.
