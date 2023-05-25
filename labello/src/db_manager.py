from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import mapper, sessionmaker, declarative_base
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.sql.expression import func
from sqlalchemy import delete
from sqlalchemy.orm.exc import NoResultFound

import models


class db_manager:
    def __init__(self):
        self.engine = create_engine("mysql://root:mysqlpass@127.0.0.1/labello_db_v2",isolation_level="READ UNCOMMITTED")
        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        session = sessionmaker()
        session.configure(bind=self.engine)
        self.session = session()

    def create_tables(self):
        """ Create tables in database """
        models.Base.metadata.create_all(self.engine)

    ############# COLOR #################################
    def add_new_color(self, col_name, col_hex_value=""):
        """ ADD COLOR """
        color_obj = models.Color()
        color_obj.col_name = col_name
        color_obj.col_hex_value = col_hex_value
        self.session.add(color_obj)
        self.session.commit()

    def get_list_colors(self):
        """ UPDATE COLOR """
        color_res = self.session.query(models.Color).all()
        lst = []
        for e in color_res:
            dico = list()
            dico.append(e.col_name)
            dico.append(e.col_hex_value)
            lst.append(dico)
        return lst

    # ############# Word #################################
    # def add_new_word(self, word_name, word_description='', type_of_word='n'):
    #     """ ADD WORD """
    #     word_obj = models.Word()
    #     word_obj.word_name = word_name
    #     word_obj.word_description = word_description
    #     word_obj.type_of_word = type_of_word
    #     self.session.add(word_obj)
    #     self.session.commit()

    # def get_list_word(self):
    #     """ UPDATE WORD """
    #     word_res = self.session.query(models.Word).all()
    #     lst = []
    #     for e in word_res:
    #         dico = list()
    #         dico.append(e.word_name)
    #         dico.append(e.word_description)
    #         dico.append(e.type_of_word)
    #         lst.append(dico)
    #     return lst
    
    # ############# Synset #################################
    # def add_new_synset(self, synset_num, synset_description=''):
    #     """ ADD SYNSET """
    #     synset_obj = models.Synset()
    #     synset_obj.synset_num = synset_num
    #     synset_obj.synset_description = synset_description
    #     self.session.add(synset_obj)
    #     self.session.commit()

    # def get_list_synset(self):
    #     """ UPDATE SYNSET """
    #     synset_res = self.session.query(models.Synset).all()
    #     lst = []
    #     for e in synset_res:
    #         dico = list()
    #         dico.append(e.synset_num)
    #         dico.append(e.synset_description)
    #         lst.append(dico)
    #     return lst
    

    ############# CATEGORIE #################################
    def add_new_categorie(self, cat_name, cat_description=''):
        """ ADD CATEGORIE """
        categorie_obj = models.Categorie()
        categorie_obj.cat_name = cat_name
        categorie_obj.cat_description = cat_description
        self.session.add(categorie_obj)
        self.session.commit()

    def get_list_categories(self):
        """ UPDATE COLOR """
        categories_res = self.session.query(models.Categorie).all()
        lst = []
        for e in categories_res:
            dico = list()
            dico.append(e.cat_name)
            dico.append(e.cat_description)
            lst.append(dico)
        return lst

    ############# SUBCATEGORIE #################################
    def add_new_subcategorie(self, scat_cat_name, scat_name, scat_description=''):
        """ ADD SUBCATEGORIE """
        subCategorie_obj = models.SubCategorie()
        subCategorie_obj.scat_name = scat_name
        subCategorie_obj.scat_cat_name = scat_cat_name
        subCategorie_obj.scat_description = scat_description
        self.session.add(subCategorie_obj)
        self.session.commit()

    def get_list_subcategories(self, p_scat_cat_name):
        """ UPDATE COLOR """
        subCategories_res = self.session.query(models.SubCategorie).filter(
            models.SubCategorie.scat_cat_name == p_scat_cat_name).all()
        lst = []
        for e in subCategories_res:
            dico = list()
            dico.append(e.scat_name)
            dico.append(e.scat_description)
            lst.append(dico)
        return lst

    ############# FILLING #################################
    def add_new_filing(self, fil_name, fil_description=''):
        """ ADD FILLING """
        filling_obj = models.Filling()
        filling_obj.fil_name = fil_name
        filling_obj.fil_description = fil_description
        self.session.add(filling_obj)
        self.session.commit()

    def get_list_fillings(self):
        """ UPDATE FILLING """
        filling_res = self.session.query(models.Filling).all()
        lst = []
        for e in filling_res:
            dico = list()
            dico.append(e.fil_name)
            dico.append(e.fil_description)
            lst.append(dico)
        return lst

    ############# MATERIAL #################################
    def add_new_material(self, mat_name, mat_description=''):
        """ ADD MATERIAL """
        material_obj = models.Material()
        material_obj.mat_name = mat_name
        material_obj.mat_description = mat_description
        self.session.add(material_obj)
        self.session.commit()

    def get_list_material(self):
        """ UPDATE MATERIAL """
        material_res = self.session.query(models.Material).all()
        lst = []
        for e in material_res:
            dico = list()
            dico.append(e.mat_name)
            dico.append(e.mat_description)
            lst.append(dico)
        return lst

    ############# MOUVABLE #################################
    def add_new_movable(self, mov_name, mov_description=''):
        """ ADD MOVABLE """
        movable_obj = models.Mouvable()
        movable_obj.mov_name = mov_name
        movable_obj.mov_description = mov_description
        self.session.add(movable_obj)
        self.session.commit()

    def get_list_movable(self):
        """ UPDATE MOVABLE """
        movable_res = self.session.query(models.Mouvable).all()
        lst = []
        for e in movable_res:
            dico = list()
            dico.append(e.mov_name)
            dico.append(e.mov_description)
            lst.append(dico)
        return lst

    ############# ROUGHNESS #################################
    def add_new_roughness(self, rug_name, rug_description=''):
        """ ADD ROUGHNESS """
        roughness_obj = models.Roughness()
        roughness_obj.rug_name = rug_name
        roughness_obj.rug_description = rug_description
        self.session.add(roughness_obj)
        self.session.commit()

    def get_list_roughness(self):
        """ UPDATE ROUGHNESS """
        roughness_res = self.session.query(models.Roughness).all()
        lst = []
        for e in roughness_res:
            dico = list()
            dico.append(e.rug_name)
            dico.append(e.rug_description)
            lst.append(dico)
        return lst

    ############# SHAPE #################################
    def add_new_shape(self, shp_name, shp_description=''):
        """ ADD ROUGHNESS """
        shape_obj = models.Shape()
        shape_obj.shp_name = shp_name
        shape_obj.shp_description = shp_description
        self.session.add(shape_obj)
        self.session.commit()

    def get_list_shape(self):
        """ UPDATE ROUGHNESS """
        shape_res = self.session.query(models.Shape).all()
        lst = []
        for e in shape_res:
            dico = list()
            dico.append(e.shp_name)
            dico.append(e.shp_description)
            lst.append(dico)
        return lst
    

    # ############# Type of word #################################
    # def add_new_type_of_word(self, shn_name, shn_description=''): 
    #     """ ADD SHINE """
    #     shane_obj = models.Shine()
    #     shane_obj.shn_name = shn_name
    #     shane_obj.shn_description = shn_description
    #     self.session.add(shane_obj)
    #     self.session.commit()

    # def get_list_type_of_word(self):
    #     """ UPDATE SHINE """
    #     shane_res = self.session.query(models.Shine).all()
    #     lst = []
    #     for e in shane_res:
    #         dico = list()
    #         dico.append(e.shn_name)
    #         dico.append(e.shn_description)
    #         lst.append(dico)
    #     return lst

    ############# SHINE #################################
    def add_new_shine(self, shn_name, shn_description=''):
        """ ADD SHINE """
        shane_obj = models.Shine()
        shane_obj.shn_name = shn_name
        shane_obj.shn_description = shn_description
        self.session.add(shane_obj)
        self.session.commit()

    def get_list_shine(self):
        """ UPDATE SHINE """
        shane_res = self.session.query(models.Shine).all()
        lst = []
        for e in shane_res:
            dico = list()
            dico.append(e.shn_name)
            dico.append(e.shn_description)
            lst.append(dico)
        return lst

    ############# Flexible #################################
    def add_new_flexible(self, flx_name, flx_description=''):
        """ ADD Flexible """
        flexible_obj = models.Material()
        flexible_obj.flx_name = flx_name
        flexible_obj.flx_description = flx_description
        self.session.add(flexible_obj)
        self.session.commit()

    def get_list_flexible(self):
        """ UPDATE Flexible """
        flexeble_res = self.session.query(models.Flexible).all()
        lst = []
        for e in flexeble_res:
            dico = list()
            dico.append(e.flx_name)
            dico.append(e.flx_description)
            lst.append(dico)
        return lst

    ############# OBJECT_NAME #################################
    def add_new_objectName(self, obn_scat_name, obn_name , obn_description=''):
        """ ADD OBJECT_NAME """
        objectName_obj = models.ObjectName()
        objectName_obj.obn_name = obn_name
        objectName_obj.obn_scat_name = obn_scat_name
        objectName_obj.obn_description = obn_description
        self.session.add(objectName_obj)
        self.session.commit()

    def get_list_objectName(self, p_obn_scat_name):
        """ OBJECT_NAME """
        objectName_res = self.session.query(models.ObjectName).filter(
            models.ObjectName.obn_scat_name == p_obn_scat_name).all()
        lst = []
        for e in objectName_res:
            dico = list()
            dico.append(e.obn_name)
            dico.append(e.obn_description)
            lst.append(dico)
        return lst

    ############# OBJECTS #####################################
    def get_last_object(self):
        """
        get id of latest object saved in database
        """
        last_obj_id = self.session.query(func.max(models.Object.obj_id)).one()
        return last_obj_id[0]

    def get_object_by_id(self, obj_id):
        """
        SELECT * FROM object, object_name, subcategorie, categorie
        WHERE
        object.obj_obn_name = object_name.obn_name AND
        object_name.obn_scat_name = subcategorie.scat_name AND
        subcategorie.scat_cat_name = categorie.cat_name
        """
        obj = self.session.query(models.Object, models.ObjectName, models.SubCategorie, models.Categorie
                                 ).filter(
            models.Object.obj_obn_name == models.ObjectName.obn_name
        ).filter(
            models.ObjectName.obn_scat_name == models.SubCategorie.scat_name
        ).filter(
            models.SubCategorie.scat_cat_name == models.Categorie.cat_name
        ).filter(models.Object.obj_id == obj_id).one()
        res = {}
        for o in obj:
            #res = res | o.as_dict()
            res = res.copy()
            res.update(o.as_dict())
        return res

    def get_list_all_object(self):
        res = self.session.query(models.Object).all()
        lst = list()
        for e in res:
            lst.append(e.as_dict())
        return lst


    def get_next_object(self, id):
        """
        get id of next object saved in database
        """
        try:
            last_obj_id = self.session.query(models.Object.obj_id).filter( models.Object.obj_id > id).order_by(models.Object.obj_id.asc()).limit(1).one()
        except Exception as e:
            return None
        return last_obj_id[0]

    def get_previous_object(self, id):
        """
        get id of previous object saved in database
        """
        try:
            last_obj_id = self.session.query(models.Object.obj_id).filter( models.Object.obj_id < id).order_by(models.Object.obj_id.desc()).limit(1).one()
        except Exception as e:
            return None
        return last_obj_id[0]

    def add_new_object(self, properties):
        obj = models.Object()
        obj.obj_wordnet_word = properties['obj_wordnet_word']
        obj.obj_wordnet_synset = properties['obj_wordnet_synset']
        #obj.obj_wordnet_type_of_word = properties['cmbx_obj_type_of_word']
        
        obj.obj_real_name = properties['obj_real_name']
        obj.obj_real_description = properties['obj_real_description']
        #obj.obj_date_create = properties['obj_date_create']

        obj.obj_size_length_x = properties['obj_size_length_x']
        obj.obj_size_width_y = properties['obj_size_width_y']
        obj.obj_size_height_z = properties['obj_size_height_z']

        obj.obj_shine = properties['obj_shine']
        obj.obj_filling = properties['obj_filling']
        obj.obj_roughness = properties['obj_roughness']
        obj.obj_weight = properties['obj_weight']
        obj.obj_obn_name = properties['obj_obn_name']
        obj.obj_shp_name = properties['obj_shp_name']
        obj.obj_color_name_1 = properties['obj_color_name_1']
        obj.obj_color_name_2 = properties['obj_color_name_2']
        obj.obj_color_name_3 = properties['obj_color_name_3']
        obj.obj_mat_name_1 = properties['obj_mat_name_1']
        obj.obj_mat_name_2 = properties['obj_mat_name_2']
        obj.obj_mat_name_3 = properties['obj_mat_name_3']
        obj.obj_mov_name = properties['obj_mov_name']
        obj.obj_flx_name = properties['obj_flx_name']
        self.session.add(obj)
        self.session.commit()


    def update_object(self, properties):
        obj = self.session.query(models.Object).filter(models.Object.obj_id == properties['obj_id']).one()

        obj.obj_wordnet_word = properties['obj_wordnet_word']
        obj.obj_wordnet_synset = properties['obj_wordnet_synset']
        #obj.obj_wordnet_type_of_word = properties['cmbx_obj_type_of_word']

        obj.obj_real_name = properties['obj_real_name']
        obj.obj_real_description = properties['obj_real_description']
        # obj.obj_date_create = properties['obj_date_create']

        obj.obj_size_length_x = properties['obj_size_length_x']
        obj.obj_size_width_y = properties['obj_size_width_y']
        obj.obj_size_height_z = properties['obj_size_height_z']

        obj.obj_shine = properties['obj_shine']
        obj.obj_filling = properties['obj_filling']
        obj.obj_roughness = properties['obj_roughness']
        obj.obj_weight = properties['obj_weight']
        obj.obj_obn_name = properties['obj_obn_name']
        obj.obj_shp_name = properties['obj_shp_name']
        obj.obj_color_name_1 = properties['obj_color_name_1']
        obj.obj_color_name_2 = properties['obj_color_name_2']
        obj.obj_color_name_3 = properties['obj_color_name_3']
        obj.obj_mat_name_1 = properties['obj_mat_name_1']
        obj.obj_mat_name_2 = properties['obj_mat_name_2']
        obj.obj_mat_name_3 = properties['obj_mat_name_3']
        obj.obj_mov_name = properties['obj_mov_name']
        obj.obj_flx_name = properties['obj_flx_name']
        self.session.commit()

    def delete_object(self, obj_id):
        obj = self.session.query(models.Object).filter(models.Object.obj_id == obj_id).one()
        self.session.delete(obj)
        self.session.commit()


    ### Images

    def get_list_all_images(self, obj_id):
        res = self.session.query(models.Image).filter(models.Image.img_obj_id == obj_id).all()
        lst = list()
        for e in res:
            lst.append(e.as_dict())
        return lst

    def get_last_image_id(self):
        """
        get id of latest Image saved in database
        """
        last_img_id = self.session.query(func.max(models.Image.img_id)).one()
        return last_img_id[0]

    def get_image_by_id(self, img_id):
        """
        get data of image by ID
        """
        img_data = self.session.query(models.Image).filter(models.Image.img_id == img_id).one()

        dico={}
        dico['img_id'] = img_data.img_id
        dico['img_specto_data'] = img_data.img_specto_data
        return dico


    def add_new_image(self, img_properties):
        img_obj = models.Image()
        img_obj.img_obj_id = img_properties['img_obj_id']
        # img_obj.img_rgb_name = img_properties['img_rgb_name']
        # img_obj.img_rgb_type = img_properties['img_rgb_type']
        # img_obj.img_rgb_size = img_properties['img_rgb_size']
        # img_obj.img_rgb_file_size = img_properties['img_rgb_file_size']
        # img_obj.img_depth_name = img_properties['img_depth_name']
        # img_obj.img_depth_size = img_properties['img_depth_size']
        # img_obj.img_depth_type = img_properties['img_depth_type']
        # img_obj.img_depth_file_size = img_properties['img_depth_file_size']
        # img_obj.img_depth_intrinsec = img_properties['img_depth_intrinsec']
        # img_obj.img_depth_extrinsec = img_properties['img_depth_extrinsec']
        # img_obj.img_depth_turntable_deg = img_properties['img_depth_turntable_deg']
        # img_obj.img_depth_distance = img_properties['img_depth_distance']
        # img_obj.img_depth_distance = img_properties['img_pc_name']
        # img_obj.img_pc_size = img_properties['img_pc_size']
        # img_obj.img_pc_type = img_properties['img_pc_type']
        # img_obj.img_pc_file_size = img_properties['img_pc_file_size']
        # img_obj.img_pc_intrinsec = img_properties['img_pc_intrinsec']
        # img_obj.img_pc_extrinsec = img_properties['img_pc_extrinsec']
        # img_obj.img_pc_turntable_deg = img_properties['img_pc_turntable_deg']
        # img_obj.img_pc_distance = img_properties['img_pc_distance ']
        # img_obj.img_specto_data = img_properties['img_specto_data']
        # img_obj.img_specto_position = img_properties['img_specto_position']
        self.session.add(img_obj)
        self.session.commit()

    def update_image(self, img_properties):
        img_obj = self.session.query(models.Image).filter(models.Image.img_id == img_properties['img_id']).one()
        img_obj.img_obj_id = img_properties['img_obj_id']
        img_obj.img_rgb_name = img_properties['img_rgb_name']
        img_obj.img_rgb_type = img_properties['img_rgb_type']
        img_obj.img_rgb_size = img_properties['img_rgb_size']
        img_obj.img_rgb_file_size = img_properties['img_rgb_file_size']
        img_obj.img_depth_name = img_properties['img_depth_name']
        img_obj.img_depth_size = img_properties['img_depth_size']
        img_obj.img_depth_type = img_properties['img_depth_type']
        img_obj.img_depth_file_size = img_properties['img_depth_file_size']
        img_obj.img_depth_intrinsec = img_properties['img_depth_intrinsec']
        img_obj.img_depth_extrinsec = img_properties['img_depth_extrinsec']
        img_obj.img_depth_turntable_deg = img_properties['img_depth_turntable_deg']
        img_obj.img_depth_distance = img_properties['img_depth_distance']
        img_obj.img_depth_distance = img_properties['img_pc_name']
        img_obj.img_pc_size = img_properties['img_pc_size']
        img_obj.img_pc_type = img_properties['img_pc_type']
        img_obj.img_pc_file_size = img_properties['img_pc_file_size']
        img_obj.img_pc_intrinsec = img_properties['img_pc_intrinsec']
        img_obj.img_pc_extrinsec = img_properties['img_pc_extrinsec']
        img_obj.img_pc_turntable_deg = img_properties['img_pc_turntable_deg']
        img_obj.img_pc_distance = img_properties['img_pc_distance ']
        img_obj.img_specto_data = img_properties['img_specto_data']
        img_obj.img_specto_position = img_properties['img_specto_position']
        img_obj.img_specto_spectr_rgb = img_properties['img_specto_spectr_rgb']
        img_obj.img_specto_graph_rgb = img_properties['img_specto_graph_rgb']
        self.session.commit()
