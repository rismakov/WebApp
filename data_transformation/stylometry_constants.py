formality_markers = {'thus', 'ergo', 'henceforth', 'hence'}
swear_words = {'fuck', 'fucking', 'fucked'}
insulting_words = {'dumb', 'stupid', 'idiot', 'dipshit', 'retard', 'retarded', 'idiotic', 'dumbass'}
scientific_terms = {'science', 'scientific', 'evidence'}
excitment_words = {
  'passion', 'passionate', 'interest', 'interesting',
  'exciting', 'excitment', 'excite',
  'fascinating', 'fascination', 'fascinate'
}
lol_terms = {'lol', 'haha', 'hahaha', 'lmao', 'lmfao'}

INVALID_URLS = {
    'URL=http', 'krhttp', 'accidenthttp', "'15758http",
    'href=http', 'bro.http', 'url=http', '68611http', 'u=http',
    'http', "'http", 'ame=http', 'school.http'
}

LINK_TYPES = ['facepalm', 'troll', 'youtube', 'tumblr.com', 'vimeo', '.gif']

freq_punctuation_per_sentence = ['?', '!', "'"]
freq_punctuation_per_word = [',', ';']
freq_token_types_per_word = {
    'ands_per_word': ['and'],
    'exclusive_words_per_word': ['but', 'except'],
    'you_per_word': ['you', 'your'],
    'they_per_word': ['they', 'them', 'he', 'she'],
    'me_per_word': ['I', 'me', 'myself'],
    'we_per_word': ['us', 'we', 'our'],
    'formality_level': formality_markers,
    'swear_words_per_word': swear_words,
    'insults_per_word': insulting_words,
    'science_per_word': scientific_terms,
    'kindness_per_word': ['kindness', 'kind'],
    'spiritual_per_word': ['spiritual', 'spirituality'],
    'articles_per_word': ['a', 'an', 'the'],
    'sad_terms_per_word': ['sad', 'miserable', 'lonely', 'depressed', 'depression'],
    'absolutist_terms_per_word': ['nothing', 'always', 'completely', 'never'],
    'empathy_per_word': {'empathy', 'empathetic', 'empath'},
    'art_per_word': {'art', 'poetry', 'painting'},
    'passion_per_word': excitment_words,
    'haha_per_word': lol_terms,
    'relationship_terms_per_word': {'relationship', 'marriage', 'family'},
    'offensive_per_word': {'offensive', 'offend', 'offended'},
    'racist_per_word': {'racist', 'bigot', 'homophobe', 'homophobic', 'transphobic', 'sexist', 'ableist'}
}

freq_tokens_per_word = {
    'howevers_per_word': ['however'],
    'ifs_per_word': ['if'],
    'thats_per_word': ['that'],
    'mores_per_word': ['more'],
    'verys_per_word': ['very'],
    'reallys_per_word': ['really'],
    'facts_per_word': ['fact'],
    'think_per_word': ['think'],
    'thinking_per_word': {'thinking'},
    'feel_per_word': ['feel'],
    'feeling_per_word': ['feeling'],
    'believe_per_word': ['believe'],
    'know_per_word': ['know'],
    'party_per_word': ['party'],
    'logic_per_word': ['logic'],
    'left_terms_per_word': ['democrat', 'liberal', 'leftist', 'lefties'],
    'right_terms_per_word': ['republican', 'conservative', 'rightist'],
    'libertarian_per_word': ['libertarian'],
    'authorirarian_per_word': ['authoritarian'],
    'music_per_word': ['music'],
    'sorrys_per_word': ['sorry'],
    'meaning_per_word': {'meaning', 'purpose'},
    'love_per_word': {'love'},
    'people_per_word': {'people'},
    'agree_per_word': {'agree'},
    'disagree_per_word': {'disagree'}
}
