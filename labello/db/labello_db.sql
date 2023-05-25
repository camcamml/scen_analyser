CREATE TABLE characteristic(
   car_id INT AUTO_INCREMENT,
   car_name VARCHAR(255),
   car_unit VARCHAR(255),
   car_descriptions VARCHAR(255),
   car_type_value VARCHAR(255),
   car_list_values VARCHAR(50),
   PRIMARY KEY(car_id)
);

CREATE TABLE color(
   color_name VARCHAR(255),
   col_hex_value VARCHAR(50),
   PRIMARY KEY(color_name)
);

CREATE TABLE material(
   mat_name VARCHAR(255),
   mat_description VARCHAR(255),
   PRIMARY KEY(mat_name)
);

CREATE TABLE categorie(
   cat_name VARCHAR(255),
   cat_description VARCHAR(255),
   PRIMARY KEY(cat_name)
);

CREATE TABLE subcategorie(
   scat_name VARCHAR(255),
   scat_description VARCHAR(255),
   cat_name VARCHAR(255) NOT NULL,
   PRIMARY KEY(scat_name),
   FOREIGN KEY(cat_name) REFERENCES categorie(cat_name)
);

CREATE TABLE object_name(
   scat_name VARCHAR(255),
   obn_name VARCHAR(255),
   obn_description VARCHAR(255),
   PRIMARY KEY(scat_name, obn_name),
   FOREIGN KEY(scat_name) REFERENCES subcategorie(scat_name)
);

CREATE TABLE shape(
   shp_name VARCHAR(255),
   shp_descripton VARCHAR(255),
   PRIMARY KEY(shp_name)
);

CREATE TABLE parameters(
   par_id INT,
   par_location_rgb VARCHAR(1024),
   par_location_depth VARCHAR(1024),
   par_location_pc VARCHAR(1024),
   PRIMARY KEY(par_id)
);

CREATE TABLE object(
   obj_id INT AUTO_INCREMENT,
   obj_wordnet_word VARCHAR(64),
   obj_wordnet_synset INT,
   obj_wordnet_type_of_word VARCHAR(8),
   
   obj_real_name VARCHAR(255),
   obj_real_description VARCHAR(255),
   obj_date_create DATETIME,
   obj_size_length_x INT,
   obj_size_width_y INT,
   obj_size_heignt_z INT,
   obj_shine VARCHAR(255),
   obj_weight INT,
   obj_filling VARCHAR(255),
   shp_name VARCHAR(255) NOT NULL,
   scat_name VARCHAR(255) NOT NULL,
   obn_name VARCHAR(255) NOT NULL,
   mat_name VARCHAR(255),
   mat_name_1 VARCHAR(255) NOT NULL,
   color_name VARCHAR(255) NOT NULL,
   color_name_1 VARCHAR(255),
   color_name_2 VARCHAR(255),
   mat_name_2 VARCHAR(255),
   PRIMARY KEY(obj_id),
   FOREIGN KEY(shp_name) REFERENCES shape(shp_name),
   FOREIGN KEY(scat_name, obn_name) REFERENCES object_name(scat_name, obn_name),
   FOREIGN KEY(mat_name) REFERENCES material(mat_name),
   FOREIGN KEY(mat_name_1) REFERENCES material(mat_name),
   FOREIGN KEY(color_name) REFERENCES color(color_name),
   FOREIGN KEY(color_name_1) REFERENCES color(color_name),
   FOREIGN KEY(color_name_2) REFERENCES color(color_name),
   FOREIGN KEY(mat_name_2) REFERENCES material(mat_name)
);

CREATE TABLE image(
   obj_id INT,
   img_id INT AUTO_INCREMENT,
   img_rgb_name VARCHAR(255),
   img_rgb_type VARCHAR(255),
   img_rgb_size VARCHAR(255),
   img_rgb_file_size INT,
   img_depth_name VARCHAR(255),
   img_depth_size VARCHAR(255),
   img_depth_type VARCHAR(255),
   img_depth_file_size INT,
   img_depth_intrinsec VARCHAR(512),
   img_depth_extrinsec VARCHAR(512),
   img_depth_turntable_deg INT,
   img_depth_distance VARCHAR(50),
   img_pc_name VARCHAR(255),
   img_pc_size VARCHAR(50),
   img_pc_type VARCHAR(50),
   img_pc_file_size VARCHAR(50),
   img_pc_intrinsec VARCHAR(512),
   img_pc_extrinsec VARCHAR(512),
   img_pc_turntable_deg VARCHAR(50),
   img_pc_distance VARCHAR(50),
   PRIMARY KEY(img_id),
   FOREIGN KEY(obj_id) REFERENCES object(obj_id)
);

CREATE TABLE object_characteristic(
   obj_id INT,
   car_id INT,
   obj_car_value VARCHAR(255),
   PRIMARY KEY(obj_id, car_id),
   FOREIGN KEY(obj_id) REFERENCES object(obj_id),
   FOREIGN KEY(car_id) REFERENCES characteristic(car_id)
);
