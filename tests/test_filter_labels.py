"""label_for: canonical slug+değer -> kullanıcıya gösterilecek Türkçe etiket.

WHY: M5 kartları ve match çipleri retriever'ın `filters` çıktısını gösterir; bu
çıktı slug'lar (heating_type='kombi_dogalgaz') ve bool'lar (has_elevator=True)
içerir. Kullanıcı 'kombi_dogalgaz' değil 'Kombi Doğalgaz', True değil 'Asansör'
görmeli. Çeviri registry'nin tek doğru kaynağından (labels + values) türetilmeli;
elle ikinci bir sözlük tutmak Gen2 modalite-karışıklığı hatasını tekrarlar.
"""

from schema.emlakjet_filters import label_for


def test_enum_value_maps_back_to_its_canonical_turkish_label():
    # heating_type enum: values {"Kombi Doğalgaz": "kombi_dogalgaz", ...}
    assert label_for("heating_type", "kombi_dogalgaz") == "Kombi Doğalgaz"
    assert label_for("room_count", "2+1") == "2+1"


def test_multi_enum_option_maps_back_to_its_label():
    # balcony_type multi_enum: composer tek seçeneği skaler olarak da yazar.
    assert label_for("balcony_type", "acik_balkon") == "Açık Balkon"


def test_bool_true_renders_the_feature_name_not_the_word_true():
    assert label_for("has_elevator", True) == "Asansör"
    assert label_for("near_metro", True) == "Metro"


def test_bool_false_is_not_surfaced_as_a_positive_feature():
    # Kart/çip yalnız mevcut özellikleri pozitif gösterir; False bir çip değildir.
    assert label_for("has_elevator", False) is None


def test_plain_string_and_int_values_pass_through_as_text():
    assert label_for("district", "Kadikoy") == "Kadikoy"
    assert label_for("gross_size_m2", 90) == "90"


def test_unknown_slug_or_none_value_returns_none():
    assert label_for("not_a_real_slug", "x") is None
    assert label_for("heating_type", None) is None
