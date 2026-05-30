from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FilterSpec:
    slug: str
    group: str
    value_type: str
    labels: tuple[str, ...]
    sources: tuple[str, ...]
    values: dict[str, str] = field(default_factory=dict)


_TR_ASCII = str.maketrans("çğıöşüâîûÇĞİÖŞÜI", "cgiosuaiucgiosui")


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label).strip().translate(_TR_ASCII).lower())


def _enum(slug: str, label: str, values: tuple[str, ...], *, sources: tuple[str, ...] = ("listing_info",)) -> FilterSpec:
    return FilterSpec(slug, "structured", "enum", (label,), sources, {value: normalize_label(value).replace(" ", "_") for value in values})


def _bool(slug: str, label: str, group: str, *, image: bool = False) -> FilterSpec:
    sources = ("property_feature", "description_llm") + (("image_vlm",) if image else ())
    return FilterSpec(slug, group, "bool", (label,), sources)


STRUCTURED_FILTERS = (
    FilterSpec("city", "structured", "str", ("İl",), ("listing_json",)),
    FilterSpec("district", "structured", "str", ("İlçe",), ("listing_json",)),
    FilterSpec("neighborhood", "structured", "str", ("Semt / Mahalle",), ("listing_json",)),
    FilterSpec("search_keyword", "structured", "str", ("Arama Kelimesi",), ()),
    FilterSpec("price_tl", "structured", "int", ("Fiyat",), ("listing_json",)),
    FilterSpec("gross_size_m2", "structured", "int", ("Brüt Metrekare",), ("listing_info",)),
    _enum("room_count", "Oda Sayısı", ("Stüdyo", "1", "1+1", "1.5+1", "2+0", "2+1", "2.5+1", "2+2", "3+0", "3+1", "3.5+1", "3+2", "4+0", "4+1", "4.5+1", "4+2", "4+3", "4+4", "5+0", "5+1", "5+2", "5+3", "5+4", "6+1", "6+2", "6+3", "6+4", "7+1", "7+2", "7+3", "8+1", "8+2", "8+3", "8+4", "9+")),
    _enum("building_age", "Binanın Yaşı", ("0 (Yeni)", "1", "2", "3", "4", "5-10", "11-15", "16-20", "21 ve üzeri")),
    _enum("floor", "Bulunduğu Kat", ("Giriş ve Alt Katlar", "Bahçe dublex", "Bahçe katı", "Düz Giriş (Zemin)", "Yüksek giriş", "Kot 1 (-1)", "Kot 2 (-2)", "Kot 3 (-3)", "Kot 4 (-4)", "Müstakil", "Bodrum Kat", "Villa tipi", "Üst Katlar", "Çatı Katı", "Çatı Dubleks", *(str(value) for value in range(1, 41)), "40+")),
    _enum("total_floors", "Binanın Kat Sayısı", (*(str(value) for value in range(1, 30)), "30 ve üzeri")),
    _enum("heating_type", "Isıtma Tipi", ("Isıtma yok", "Doğalgaz sobalı", "Güneş Enerjisi", "Jeotermal", "Merkezi Doğalgaz", "Merkezi Fueloil", "Merkezi Kömür", "Merkezi (Pay Ölçer)", "Kat Kaloriferi", "Klimalı", "Kombi Doğalgaz", "Kombi Fueloil", "Kombi Katı Yakıt", "Kombi Kömür", "Sobalı", "Yerden ısıtma", "Isı Pompası", "Şömine", "Fancoil Ünitesi", "VRV", "Elektrikli Radyatör")),
    _enum("bathroom_count", "Banyo Sayısı", ("Yok", "1", "2", "3", "4", "5", "6+")),
    _enum("building_condition", "Yapı Durumu", ("Sıfır", "İkinci El", "Yapım Aşamasında")),
    _enum("occupancy", "Kullanım Durumu", ("Boş", "Kiracı Oturuyor", "Mülk Sahibi Oturuyor")),
    _enum("title_deed_status", "Tapu Durumu", ("Arsa Tapulu", "Hisseli Tapu", "Kat Mülkiyeti", "Kat İrtifakı", "Müstakil Tapulu", "Yabancıdan", "Tapu Kaydı Yok", "Kıbrıs Tapulu", "Kooperatiften Tapu", "Bilinmiyor")),
    FilterSpec("has_virtual_tour", "structured", "bool", ("Görüntülü Gezilebilir mi?",), ("listing_info",)),
    FilterSpec("is_furnished", "structured", "bool", ("Eşya Durumu",), ("listing_info", "description_llm")),
    FilterSpec("has_balcony", "structured", "bool", ("Balkon Durumu",), ("listing_info", "description_llm", "image_vlm")),
    FilterSpec("balcony_type", "structured", "multi_enum", ("Balkon Tipi",), ("listing_info", "description_llm", "image_vlm"), {
        "Açık Balkon": "acik_balkon", "Açık Teras": "acik_teras", "Fransız Balkon": "fransiz_balkon",
        "Kapalı Balkon": "kapali_balkon", "Kapalı Teras": "kapali_teras",
    }),
    FilterSpec("in_gated_complex", "structured", "bool", ("Site İçerisinde",), ("listing_info", "description_llm")),
    _enum("seller_type", "Kimden", ("Emlak Ofisinden", "Sahibinden", "Müteahhitten")),
    _enum("listing_age", "İlan Tarihi", ("Son 24 Saat", "Son 3 Gün", "Son 7 Gün", "Son 15 Gün", "Son 30 Gün"), sources=()),
)


_FEATURE_GROUPS = {
    "ic_ozellikler.altyapi": (
        ("has_adsl", "ADSL"), ("has_smart_home", "Akıllı Ev"), ("has_fiber", "Fiber"), ("has_intercom", "Intercom"),
        ("has_cable_tv_satellite", "Kablo TV - Uydu"), ("has_wifi", "Wi-Fi"), ("has_face_fingerprint_access", "Yüz Tanıma & Parmak İzi"),
    ),
    "ic_ozellikler.banyo": (
        ("has_squat_toilet", "Alaturka Tuvalet"), ("has_washing_machine", "Çamaşır Makinesi"), ("has_dryer", "Çamaşır Kurutma Makinesi"),
        ("has_shower_cabin", "Duşakabinli"), ("has_ensuite_bathroom", "Ebeveyn Banyo"), ("has_hilton_bathroom", "Hilton Banyo"),
        ("has_italian_bathroom", "İtalyan Banyo"), ("has_jacuzzi", "Jakuzi"), ("has_bathtub", "Küvet"), ("has_sauna", "Sauna"),
        ("has_hot_water", "Sıcak Su"), ("has_water_heater", "Şofben"),
    ),
    "ic_ozellikler.dekorasyon": (
        ("has_white_goods", "Beyaz Eşya"), ("has_laundry_room", "Çamaşır Odası"), ("has_steel_door", "Çelik Kapı"), ("has_wallpaper", "Duvar Kağıdı"),
        ("has_dressing_room", "Giyinme Odası"), ("has_builtin_wardrobe", "Gömme Dolap"), ("has_cornice", "Kartonpiyer"), ("has_aircon", "Klima"),
        ("has_laminate_floor", "Laminant"), ("has_panel_door", "Panel Kapı"), ("has_shutter", "Panjur"), ("has_parquet_floor", "Parke"),
        ("has_ceramic_floor", "Seramik Zemin"), ("has_spotlight", "Spot Işık"), ("has_coatroom", "Vestiyer"), ("has_high_ceiling", "Yüksek Tavan"),
    ),
    "ic_ozellikler.mutfak": (
        ("has_american_kitchen", "Amerikan Mutfak"), ("has_builtin_kitchen", "Ankastre Mutfak"), ("has_dishwasher", "Bulaşık Makinesi"),
        ("has_fridge", "Buzdolabı"), ("has_kitchen_cabinets", "Dolaplı Mutfak"), ("has_oven", "Fırın"), ("has_ready_kitchen", "Hazır Mutfak"),
        ("has_italian_kitchen", "İtalyan Mutfak"), ("has_pantry", "Kiler"), ("has_lacquer_kitchen", "Lake Mutfak"), ("has_laminate_kitchen", "Laminant Mutfak"),
        ("has_marley", "Marley"), ("has_kitchen_furniture", "Mutfak Mobilyası"), ("has_natural_gas_stove", "Ocak Doğalgazı"), ("has_countertop_stove", "Setüstü Ocak"),
    ),
    "dis_ozellikler.bina": (
        ("has_wood_joinery", "Ahşap Doğrama"), ("has_aluminum_joinery", "Alüminyum Doğrama"), ("has_caretaker", "Apartman Görevlisi"),
        ("has_elevator", "Asansör"), ("has_private_garden", "Bahçe - Müstakil"), ("has_shared_garden", "Bahçe - Ortak"), ("has_video_intercom", "Görüntülü Diafon"),
        ("has_security", "Güvenlik"), ("has_burglar_alarm", "Hırsız Alarmı"), ("has_hydrophore", "Hidrofor"), ("has_double_glazing", "Isıcam"),
        ("has_thermal_insulation", "Isı Yalıtımı"), ("has_generator", "Jeneratör"), ("has_camera_system", "Kamera Sistemi"), ("has_pvc_joinery", "PVC Doğrama"),
        ("has_sound_insulation", "Ses Yalıtımı"), ("has_siding", "Siding"), ("has_water_tank", "Su Deposu"), ("has_fireplace", "Şömine"),
        ("has_fire_alarm", "Yangın Alarmı"), ("has_fire_escape", "Yangın Merdiveni"),
    ),
    "dis_ozellikler.cephe": (
        ("has_west_facade", "Batı Cepheli"), ("has_east_facade", "Doğu Cepheli"), ("has_south_facade", "Güney Cepheli"), ("has_north_facade", "Kuzey Cepheli"),
    ),
    "dis_ozellikler.sosyal_imkanlar": (
        ("has_outdoor_pool", "Açık Havuz"), ("has_open_parking", "Açık Otopark"), ("near_shopping_mall", "Alışveriş Merkezi"), ("has_garden", "Bahçe"),
        ("has_basketball_court", "Basketbol Sahası"), ("has_playground", "Çocuk Parkı"), ("is_accessible", "Engelliye Uygun"), ("has_fitness", "Fitness"),
        ("has_football_field", "Futbol Sahası"), ("has_hammam", "Hamam"), ("has_indoor_pool", "Kapalı Havuz"), ("has_closed_parking", "Kapalı Otopark"),
        ("has_private_pool", "Müstakil Havuzlu"), ("has_tennis_court", "Tenis Kortu"), ("has_volleyball_court", "Voleybol Sahası"), ("has_walking_track", "Yürüyüş Parkuru"),
    ),
    "konum_ozellikleri.manzara": (
        ("has_bosphorus_view", "Boğaz Manzaralı"), ("has_sea_view", "Deniz Manzaralı"), ("has_lake_view", "Göl Manzaralı"),
        ("has_pool_view", "Havuz Manzaralı"), ("has_city_view", "Şehir Manzaralı"), ("has_green_view", "Yeşil Alan Manzaralı"),
    ),
    "konum_ozellikleri.ulasim": (
        ("near_main_road", "Anayol"), ("near_eurasia_tunnel", "Avrasya Tüneli"), ("near_bosphorus_bridges", "Boğaz Köprüleri"),
        ("near_street", "Caddeye Yakın"), ("near_mosque", "Camiye Yakın"), ("near_sea_bus", "Deniz Otobüsü"), ("seafront", "Denize Sıfır"),
        ("near_sea", "Denize Yakın"), ("near_dolmus", "Dolmuş"), ("near_e5", "E-5"), ("near_hospital", "Hastaneye Yakın"),
        ("near_airport", "Havaalanı"), ("near_marmaray", "Marmaray"), ("near_metro", "Metro"), ("near_metrobus", "Metrobüs"), ("near_minibus", "Minibüs"),
        ("near_school", "Okula Yakın"), ("near_highway", "Otoban"), ("near_bus", "Otobüs"), ("near_bazaar", "Semt Pazarına Yakın"),
        ("near_cable_car", "Teleferik"), ("near_tem", "TEM"), ("near_tram", "Tramvay"), ("near_train_station", "Tren İstasyonu"),
        ("near_trolleybus", "Troleybüs"), ("near_ferry", "Vapur İskelesi"),
    ),
}

_IMAGE_SLUGS = {
    "has_aircon", "has_elevator", "has_shower_cabin", "has_bathtub", "has_jacuzzi",
    "has_open_parking", "has_closed_parking", "has_outdoor_pool", "has_indoor_pool",
    "has_private_pool", "has_garden", "has_private_garden", "has_shared_garden",
    "has_playground", "has_fitness", "has_basketball_court", "has_football_field",
    "has_tennis_court", "has_volleyball_court", "has_sea_view", "has_bosphorus_view",
    "has_lake_view", "has_pool_view", "has_city_view", "has_green_view", "has_american_kitchen",
}

PROPERTY_FEATURE_SPECS = tuple(
    _bool(slug, label, group, image=slug in _IMAGE_SLUGS)
    for group, entries in _FEATURE_GROUPS.items()
    for slug, label in entries
)
EMLAKJET_FILTERS = STRUCTURED_FILTERS + PROPERTY_FEATURE_SPECS

_INFO_BY_LABEL = {
    normalize_label(label): spec
    for spec in STRUCTURED_FILTERS
    if "listing_info" in spec.sources
    for label in spec.labels
}
_PROPERTY_FEATURE_BY_LABEL = {
    normalize_label(label): spec
    for spec in PROPERTY_FEATURE_SPECS
    for label in spec.labels
}


def spec_for_info_label(label: str) -> FilterSpec | None:
    return _INFO_BY_LABEL.get(normalize_label(label))


def spec_for_property_feature(label: str) -> FilterSpec | None:
    return _PROPERTY_FEATURE_BY_LABEL.get(normalize_label(label))


def empty_filter_values() -> dict[str, Any]:
    return {spec.slug: None for spec in EMLAKJET_FILTERS}
