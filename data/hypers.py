TOXIC_EXTRACTED_ATTRS_WILDS = {
    "gender": [
        "male",
        "female",
        "transgender",
        "other_gender",
    ],
    "orientation": [
        "heterosexual",
        "homosexual_gay_or_lesbian",
        "bisexual",
        "other_sexual_orientation",
    ],
    "religion": [
        "christian",
        "jewish",
        "muslim",
        "hindu",
        "buddhist",
        "atheist",
        "other_religion",
    ],
    "race": ["black", "white", "asian", "latino", "other_race_or_ethnicity"],
    "disability": [
        "physical_disability",
        "intellectual_or_learning_disability",
        "psychiatric_or_mental_illness",
        "other_disability",
    ],
}

TOXIC_EXTRACTED_ATTRS_RACE = {
    "black": ["black"],
    "white": ["white"],
    "asian": ["asian"],
    "other_race": ["latino", "other_race_or_ethnicity"],
}

TOXIC_EXTRACTED_ATTRS_RELIGION = {
    "christian": ["christian"],
    "jewish": ["jewish"],
    "muslim": ["muslim"],
    "other_minus_religion": ["hindu", "buddhist", "atheist", "other_religion"],
}

TOXIC_EXTRACTED_ATTRS_GENDER = {
    "male": ["male"],
    "female": ["female"],
    "special": [
        "homosexual_gay_or_lesbian",
        "bisexual",
        "other_sexual_orientation",
        "transgender",
        "other_gender",
    ],
}

TOXIC_EXTRACTED_ATTRS_MIX = {
    "male": ["male"],
    "female": ["female"],
    "special_gender": [
        "homosexual_gay_or_lesbian",
        "bisexual",
        "other_sexual_orientation",
        "transgender",
        "other_gender",
    ],
    "black": ["black"],
    "white": ["white"],
    "asian": ["asian"],
    "other_race": ["latino", "other_race_or_ethnicity"],
    "christian": ["christian"],
    "jewish": ["jewish"],
    "muslim": ["muslim"],
    "other_minus_religion": ["hindu", "buddhist", "atheist", "other_religion"],
}

EXTRACTED_ATTRS_MAPPING = {
    "race": TOXIC_EXTRACTED_ATTRS_RACE,
    "gender": TOXIC_EXTRACTED_ATTRS_GENDER,
    "religion": TOXIC_EXTRACTED_ATTRS_RELIGION,
    "mix": TOXIC_EXTRACTED_ATTRS_MIX,
    "wilds": TOXIC_EXTRACTED_ATTRS_WILDS,
}

EXTRACTED_ATTRS_TASKNUM_MAPPING = {
    "race": 5,
    "gender": 4,
    "religion": 5,
    "mix": 12,
    "wilds": 6,
}


CALI_PARAMS = {
    "n_bins": 10,
    "interpolate_kind": "linear",  
}
