from db_manager import *
from Params import *

class Scan_object:

    def __init__(self):

        self.db_manager = db_manager()
        self.properties = {}
        self.list_image = []
        self.current_image = {}

    def init_to_last_object(self):
        last_id = self.db_manager.get_last_object()
        if last_id is None:
            self.properties = {}
            return
        self.properties = self.db_manager.get_object_by_id(last_id)

    def set_properties_value(self, property_name, new_value):
        self.properties[property_name] = new_value


    def next_object(self):
        """ Move to next object """
        id_next = self.db_manager.get_next_object(self.properties["obj_id"])
        if id_next is None:
            return
        self.properties = self.db_manager.get_object_by_id(id_next)

    def previous_object(self):
        """ move to the previous object  """
        id_previous = self.db_manager.get_previous_object(self.properties["obj_id"])
        if id_previous is None:
            return
        self.properties = self.db_manager.get_object_by_id(id_previous)

    def move_to_object(self, obj_id):
        """ move to the  object by id  """
        self.properties = self.db_manager.get_object_by_id(obj_id)
        pass

    def get_object_list(self):
        """ get list of all objects"""
        lst = []
        for elem in self.db_manager.get_list_all_object():
            lst.append(list(elem.values()))
        return lst

    def add_color(self, properties):
        """ add color """
        pass

    def get_list_color(self):
        """ get list of colors """
        lst = self.db_manager.get_list_colors()
        return lst

    def add_material(self, properties):
        """ add color """
        pass

    def get_list_material(self):
        """ get list of colors """
        lst = self.db_manager.get_list_material()
        return lst

    def add_categorie(self, cat_name, cat_description):
        """ add categories """
        self.db_manager.add_new_categorie(cat_name, cat_description);
        return

    def add_sub_categorie(self, scat_name, scat_cat_name, scat_description=''):
        """ add categories """
        self.db_manager.add_new_subcategorie (scat_name, scat_cat_name, scat_description='')
        return

    def add_object_name(self, obn_scat_name, obn_name, obn_description=''):
        """ add categories """
        self.db_manager.add_new_objectName(obn_scat_name, obn_name, obn_description='')
        return


    def add_type_of_word(self, properties):
        """ add type_of_word"""
        self.db_manager.add_new_type_of_word(properties);
        return

    def get_list_type_of_word(self):
        lst = self.db_manager.get_list_type_of_word()
        return lst



    def get_list_categories(self):
        """ get list of colors """
        lst = self.db_manager.get_list_categories()
        return lst

    def add_flexible(self, properties):
        """ add flexible """
        self.db_manager.add_new_flexible(properties);
        return

    def get_list_flexible(self):
        """ get list of flexible """
        lst = self.db_manager.get_list_flexible()
        return lst

    def add_shine(self, properties):
        """ add categorie """
        self.db_manager.add_new_shine(properties);
        return

    def get_list_shine(self):
        """ get list of colors """
        lst = self.db_manager.get_list_shine()
        return lst

    def add_filling(self, properties):
        """ add filling """
        self.db_manager.add_new_filing(properties);
        return

    def get_list_filling(self):
        """ get list of filling """
        lst = self.db_manager.get_list_fillings()
        return lst

    def add_movable(self, properties):
        """ add movable """
        self.db_manager.add_new_filing(properties);
        return

    def get_list_movable(self):
        """ get list of movable """
        lst = self.db_manager.get_list_movable()
        return lst

    def get_list_subcategories(self, id_categorie):
        """ get list of subcategories """
        lst = self.db_manager.get_list_subcategories(id_categorie)
        return lst

    def get_list_object_name(self, id_subcategorie):
        """ get list of name """
        lst = self.db_manager.get_list_objectName(id_subcategorie)
        return lst

    def add_material(self, properties):
        """ add material """
        pass

    def add_object(self):
        self.db_manager.add_new_object(self.properties)
        self.init_to_last_object()

    def save_curent_object(self):
        """ save curent object  """
        self.db_manager.update_object(self.properties)
        self.move_to_object(self.properties["obj_id"])

    def delete_curent_object(self):
        """ delete curent object  """
        self.db_manager.delete_object(self.properties['obj_id'])
        self.next_object()

    def get_current_object_prop(self):
        """ return dictionaty with object proprty """
        ## get object
        return self.properties

    def get_list_images(self):
        """ get list of images of curent object
            return py list of list  with featurs of images
            [
              [rgb images],
              [depth images],
              [spectroscop data]
            ]
        """
        lst = []
        for elem in self.db_manager.get_list_all_images(self.properties['obj_id']):
            lst.append(list(elem.values()))
        return lst


    def create_new_images(self, data):
        self.current_image['img_obj_id'] = self.properties['obj_id']
        self.db_manager.add_new_image(self.current_image)

        img_id = self.db_manager.get_last_image_id()
        self.current_image['img_id'] = img_id
        rgb_img_name = "rgb_" + str(self.properties['obj_id']) + '_' + str(img_id) + ".jpg"
        depth_name = "depth_"  + str(self.properties['obj_id']) + '_' + str(img_id) + ".png"
        rgb_specrto_name = "spectro_" + str(self.properties['obj_id']) + '_' + str(img_id) + ".png"
        rgb_graph_name = "graph_" + str(self.properties['obj_id']) + '_' + str(img_id) + ".png"


        self.current_image['img_rgb_name'] = rgb_img_name
        self.current_image['img_rgb_type'] = ""
        self.current_image['img_rgb_size'] = 0
        self.current_image['img_rgb_file_size'] = 0
        self.current_image['img_depth_name'] = depth_name
        self.current_image['img_depth_size'] = 0
        self.current_image['img_depth_type'] = ""
        self.current_image['img_depth_file_size'] = 0
        self.current_image['img_depth_intrinsec'] = data['img_depth_intrinsec']
        self.current_image['img_depth_extrinsec'] = data['img_depth_extrinsec']
        self.current_image['img_depth_turntable_deg'] = 0
        self.current_image['img_depth_distance'] = ""
        self.current_image['img_pc_name'] = ""
        self.current_image['img_pc_size'] = ""
        self.current_image['img_pc_type'] = ""
        self.current_image['img_pc_file_size'] = ""
        self.current_image['img_pc_intrinsec'] = data['img_pc_intrinsec']
        self.current_image['img_pc_extrinsec'] = data['img_pc_extrinsec']
        self.current_image['img_pc_turntable_deg'] = 0
        self.current_image['img_pc_distance '] = 0
        self.current_image['img_specto_data'] =  data["img_specto_data"]
        self.current_image['img_specto_position'] = 0
        self.current_image['img_specto_spectr_rgb'] = rgb_specrto_name
        self.current_image['img_specto_graph_rgb'] = rgb_graph_name
        self.db_manager.update_image(self.current_image)
        return self.current_image

    def load_image(self, obj_id, img_id):
        self.current_image['img_obj_id'] = obj_id
        self.current_image['img_id'] = img_id
        rgb_img_name = "rgb_" + str(obj_id) + '_' + str(img_id) + ".jpg"
        depth_name = "depth_"  + str(obj_id) + '_' + str(img_id) + ".png"
        rgb_specrto_name = "spectro_" + str(obj_id) + '_' + str(img_id) + ".png"
        rgb_graph_name = "graph_" + str(self.properties['obj_id']) + '_' + str(img_id) + ".png"


        data_img = self.db_manager.get_image_by_id(img_id)

        self.current_image['img_rgb_name'] = rgb_img_name
        self.current_image['img_rgb_type'] = ""
        self.current_image['img_rgb_size'] = 0
        self.current_image['img_rgb_file_size'] = 0
        self.current_image['img_depth_name'] = depth_name
        self.current_image['img_depth_size'] = 0
        self.current_image['img_depth_type'] = ""
        self.current_image['img_depth_file_size'] = 0
        self.current_image['img_depth_intrinsec'] = ""
        self.current_image['img_depth_extrinsec'] = ""
        self.current_image['img_depth_turntable_deg'] = 0
        self.current_image['img_depth_distance'] = ""
        self.current_image['img_pc_name'] = ""
        self.current_image['img_pc_size'] = ""
        self.current_image['img_pc_type'] = ""
        self.current_image['img_pc_file_size'] = ""
        self.current_image['img_pc_intrinsec'] = ""
        self.current_image['img_pc_extrinsec'] = ""
        self.current_image['img_pc_turntable_deg'] = 0
        self.current_image['img_pc_distance '] = 0
        self.current_image['img_specto_data'] = data_img['img_specto_data']
        self.current_image['img_specto_position'] = 0
        self.current_image['img_specto_spectr_rgb'] = rgb_specrto_name
        self.current_image['img_specto_graph_rgb'] = rgb_graph_name



