from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime, Integer, Text
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

import datetime

Base = declarative_base()

### Identification Djerroud ###
class Categorie(Base):
    __tablename__ = 'categorie'
    cat_name = Column(String(255), primary_key=True)
    cat_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class SubCategorie(Base):
    __tablename__ = 'subcategorie'
    scat_name = Column(String(255), primary_key=True)
    scat_description = Column(String(255))
    scat_cat_name = Column(String(255), ForeignKey("categorie.cat_name"))
    relationship("categorie")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ObjectName(Base):
    __tablename__ = 'object_name'
    obn_name = Column(String(255), primary_key=True)
    obn_description = Column(String(255))
    obn_scat_name = Column(String(255), ForeignKey("subcategorie.scat_name"))
    relationship("subcategorie")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


### Identification NLTK Wordnet ###
class Word(Base):
    __tablename__ = 'word'
    word_name = Column(String(255), primary_key=True)
    word_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
class Type_of_word(Base):
    __tablename__ = 'type_of_word'
    typew_name = Column(String(255), primary_key=True)
    typew_description = Column(String(255))
    typew_word_name = Column(String(255), ForeignKey("word.word_name"))
    relationship("word")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Synset(Base):
    __tablename__ = 'synset'
    synset_num = Column(Integer(), primary_key=True)
    synset_description = Column(String(255))
    synset_typew_name = Column(String(255), ForeignKey("type_of_word.typew_name"))
    relationship("type_of_word")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
### ###

class Shape(Base):
    __tablename__ = 'shape'
    shp_name = Column(String(255), primary_key=True)
    shp_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Color(Base):
    __tablename__ = 'color'
    col_name = Column(String(255), primary_key=True)
    col_hex_value = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Material(Base):
    __tablename__ = 'material'
    mat_name = Column(String(255), primary_key=True)
    mat_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Mouvable(Base):
    __tablename__ = 'mouvable'
    mov_name = Column(String(255), primary_key=True)
    mov_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Shine(Base):
    __tablename__ = 'shine'
    shn_name = Column(String(255), primary_key=True)
    shn_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Roughness(Base):
    __tablename__ = 'roughness'
    rug_name = Column(String(255), primary_key=True)
    rug_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Filling(Base):
    __tablename__ = 'filling'
    fil_name = Column(String(255), primary_key=True)
    fil_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Flexible(Base):
    __tablename__ = 'flexible'
    flx_name = Column(String(255), primary_key=True)
    flx_description = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}



class Object(Base):
    __tablename__ = 'object'
    obj_id = Column(Integer(), primary_key=True)
    obj_real_name = Column(String(255), default='')
    obj_real_description = Column(String(255), default='')
    obj_date_create= Column(DateTime(), default = datetime.datetime.utcnow())
    obj_size_length_x = Column(Integer(), default = 0)
    obj_size_width_y = Column(Integer(), default = 0)
    obj_size_height_z = Column(Integer(), default = 0)

    obj_shine = Column(String(255), ForeignKey("shine.shn_name"))
    relationship("shine")

    obj_filling = Column(String(255), ForeignKey("filling.fil_name"))
    relationship("filling")

    obj_roughness = Column(String(255), ForeignKey("roughness.rug_name"))
    relationship("roughness")

    obj_weight = Column(Integer())

    obj_obn_name = Column(String(255), ForeignKey("object_name.obn_name"))
    relationship("object_name")

    obj_shp_name = Column(String(255), ForeignKey("shape.shp_name"))
    relationship("shape")

    obj_color_name_1 = Column(String(255), ForeignKey("color.col_name"))
    obj_color_name_2 = Column(String(255), ForeignKey("color.col_name"))
    obj_color_name_3 = Column(String(255), ForeignKey("color.col_name"))
    relationship("color")

    obj_mat_name_1 = Column(String(255), ForeignKey("material.mat_name"))
    obj_mat_name_2 = Column(String(255), ForeignKey("material.mat_name"))
    obj_mat_name_3 = Column(String(255), ForeignKey("material.mat_name"))
    relationship("material")

    obj_mov_name = Column(String(255), ForeignKey("mouvable.mov_name"))
    relationship("mouvable")

    obj_flx_name = Column(String(255), ForeignKey("flexible.flx_name"))
    relationship("flexible")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class Image(Base):
    __tablename__ = 'image'
    img_id = Column(Integer(), primary_key=True)
    img_rgb_name = Column(String(255))
    img_rgb_type = Column(String(255))
    img_rgb_size = Column(String(255))
    img_rgb_file_size = Column(Integer())
    img_depth_name = Column(String(255))
    img_depth_size = Column(String(255))
    img_depth_type = Column(String(255))
    img_depth_file_size = Column(Integer())
    img_depth_intrinsec = Column(String(512))
    img_depth_extrinsec = Column(String(512))
    img_depth_turntable_deg = Column(Integer())
    img_depth_distance = Column(String(50))
    img_pc_name = Column(String(255))
    img_pc_size = Column(String(50))
    img_pc_type = Column(String(50))
    img_pc_file_size = Column(String(50))
    img_pc_intrinsec = Column(String(512))
    img_pc_extrinsec = Column(String(512))
    img_pc_turntable_deg = Column(Integer())
    img_pc_distance = Column(Integer())
    img_specto_data = Column(Text())
    img_specto_position = Column(Integer())
    img_specto_spectr_rgb = Column(String(255))
    img_specto_graph_rgb = Column(String(255))
    img_obj_id = Column(Integer(), ForeignKey("object.obj_id"))
    relationship("object")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class Characteristic(Base):
    __tablename__ = 'characteristic'
    car_id = Column(Integer(), primary_key=True)
    car_name = Column(String(255))
    car_unit = Column(String(255))
    car_descriptions = Column(String(255))
    car_type_value = Column(String(255))
    car_list_values = Column(String(255))

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ObjectCharacteristic(Base):
    __tablename__ = 'object_characteristic'
    ocar_obj_id = Column(Integer(), ForeignKey("object.obj_id"), primary_key=True)
    ocar_car_id = Column(Integer(), ForeignKey("characteristic.car_id"), primary_key=True)
    ocar_obj_value = Column(String(255))
    relationship("object")
    relationship("characteristic")

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

