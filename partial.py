user_one = {
    'name': 'Rogerio',
    'networks': [{
        'name': 'facebook',
        'prof_img': 'foto de perfil',
        'cover_img': 'foto de capa',
    },{
        'name': 'twitter',
        'prof_img': 'imagem do usuario',
        'cover_img': 'imagem do perfil',
    },]
}
user_two = {
    'name': 'hubbbertrtt',
    'networks': []
}

def get_prof_img(user):
    for network in user['networks']:
        if network['prof_img']:
            yield network['name'], network['prof_img']
    return None, None

list(get_prof_img(user_one))
# [('facebook', 'foto de perfil'), ('twitter', 'imagem do usuario')]
list(get_prof_img(user_two))
# []


def get_img(img_type, user):
    for network in user['networks']:
        if network[img_type]:
            yield network['name'], network[img_type]
    return None, None

get_prof_img = lambda user: get_img('prof_img', user)
get_cover_img = lambda user: get_img('cover_img', user)

list(get_prof_img(user_one))
# [('facebook', 'foto de perfil'), ('twitter', 'imagem do usuario')]
list(get_cover_img(user_one))
# [('facebook', 'foto de capa'), ('twitter', 'imagem do perfil')]


from functools import partial

get_prof_img = partial(get_img, 'prof_img')
get_cover_img = partial(get_img, 'cover_img')

list(get_prof_img(user_one))
# [('facebook', 'foto de perfil'), ('twitter', 'imagem do usuario')]
list(get_cover_img(user_one))
# [('facebook', 'foto de capa'), ('twitter', 'imagem do perfil')]
