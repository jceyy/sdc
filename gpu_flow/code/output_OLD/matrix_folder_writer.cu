#include "matrix_folder_writer.h"

matrix_folder* matrix_folder_writer::read_binary_file(string filename){

    ifstream ifs(filename.c_str(), ios::in | ios::binary);
    if(!ifs.is_open()) {
        EXIT_ERROR2("Open file failed:", filename.c_str());
    }

    int number_of_matrices = 0;
	ifs.read((char*) &number_of_matrices, sizeof(int));

    #ifdef DEBUG
        cout << "number of matrices: " << number_of_matrices << endl;
    #endif
    matrix_folder* output = new matrix_folder(number_of_matrices);

    //...iterate to read all matrices
	for(int i = 0; i < number_of_matrices; i++) {
		int current_matrix_dimension = 0;
		ifs.read((char*) &current_matrix_dimension, sizeof(int));

        #ifdef DEBUG
            cout << "matrix : " << i << " matrix-dimension " << current_matrix_dimension << endl;
        #endif

		//...read out the dimensions for the current matrix
        vector<int> dimensions;
		int current_dimension = 0;
		int number_of_matrix_entries = 1;
		for(int j = 0; j < current_matrix_dimension; j++) {
			ifs.read((char*) &current_dimension, sizeof(int));
			dimensions.push_back(current_dimension);
			number_of_matrix_entries *= current_dimension;

            #ifdef DEBUG
                cout << "matrix : " << i << " dim " << j  << " matrix dimension " << current_dimension << endl;
            #endif
		}

		//...read out the current data
		CUDA_FLOAT* data = new CUDA_FLOAT[number_of_matrix_entries];
		ifs.read((char*) data, sizeof(CUDA_FLOAT) * number_of_matrix_entries);		
		
		//create host matrix
		matrix_host* current_host_matrix = new matrix_host(data, dimensions); 

		//create device matrix
        matrix_device* current_device_matrix = new matrix_device(current_host_matrix, matrix_device::Copy);
		output->add_matrix(i, current_device_matrix);

        delete current_host_matrix;
	}

	//close stream and return folder
	ifs.close();
	return output;
}


void matrix_folder_writer::write_binary_file(string filename, matrix_folder* folder){

	//open binary file
	ofstream of(filename.c_str(), ios::out | ios::binary);
    if(!of.is_open()) {
        cerr << "Open file failed:" << filename << endl;
    }

	//write out some starting informations
	//...the number of matrices inside the matrix_folder
	int dim = folder->get_dimension();
	of.write((char*) &dim, sizeof(int));
	
	//...iterate over the number of matrices within the folder
	for(int i = 0; i < dim; i++) {

		//...copy current matrix to host memory
		matrix_device* current_matrix = folder->get_matrix(i);
        matrix_host* current_host_copy = new matrix_host(current_matrix, matrix_host::Copy);

		//informations about the size of the data!
		//...the number of dimensions
		int current_matrix_dimension = current_host_copy->get_dimensions();
		of.write((char*) &current_matrix_dimension, sizeof(int));
		//...loop over the dimensions
        vector<int> matrix_dimensions = current_host_copy->get_matrix_dimensions();
		int current_dim = 1;
		for(int j = 0; j < current_matrix_dimension; j++){
			current_dim = matrix_dimensions.at(j);
			of.write((char*) &current_dim, sizeof(int));
		}	

		//...write out the data
		int number_of_matrix_entries = current_host_copy->get_number_of_matrix_entries();
		CUDA_FLOAT* data = current_host_copy->get_data();
		//.......write complex data
		of.write((char*) data, sizeof(CUDA_FLOAT) * number_of_matrix_entries);	

		delete current_host_copy;
	}

	of.close();
}

/*!
* writes out the current folder in a vtl file
* @param type use 0=>imag , 1=> real part, 2=> magnitude
*/
void matrix_folder_writer::write_vtk_file(string filename, matrix_folder* folder, int type){
	ofstream of(filename.c_str());
    if(!of.is_open()) {
        cerr << "Open file failed:" << filename << endl;
    }

	//write header information
    of << "# vtk DataFile Version 3.1" << endl;
    of << "pseudo-spectral simulation" << endl;
    of << "ASCII " << endl;
    of << endl;
    //of << "DATASET POLYDATA" << endl;
    of << "DATASET UNSTRUCTURED_GRID" << endl;
    of << endl;
	of.precision(10);
	of.setf(ios::scientific);



	//write point informations
	int dim = folder->get_dimension();
	#ifdef OPTIMIZATION_DATA_TRANSFER_DEBUG
    cout << "...transfer to host memory!" << endl;
	#endif
    matrix_host* m_host = new matrix_host(folder->get_matrix(0), matrix_host::Copy);
	#ifdef OPTIMIZATION_DATA_TRANSFER_DEBUG
    cout << "...write vtk data to disk!" << endl;
	#endif
	int num_entries = m_host->get_number_of_matrix_entries();
    of << "POINTS " << num_entries << " DOUBLE" << endl;
	matrix_host_iterator* it =  matrix_host_iterator::create_iterator(m_host);
	while(it->has_next()) {
		it->next();
		vector<int> indices = it->get_current_indices();
        for(size_t k = 0; k < 3; k++) {
			if(indices.size() > k)
				of << indices.at(k) << " ";
			else
				of << "0" << " ";
		}
        of << endl;
		
	}

	//write cell data
	if(m_host->get_matrix_dimensions().size() == 2)	{
		int cols =m_host->get_matrix_dimensions().at(0);
		int rows = m_host->get_matrix_dimensions().at(1);
        of << "CELLS " << (cols-1)*(rows-1) <<" " << (cols-1)*(rows-1)*5 << endl;
		for(int j = 0; j < rows-1; j++) {
			for(int i = 0 ; i < cols-1; i++){
                of << "4 " << i + j*cols << " " << i+1+j*cols << " " << i+(j+1)*cols << " " << i+1+(j+1)*cols << endl;
			}
		} 
        of << "CELL_TYPES " << (cols-1)*(rows-1) << endl;
		for(int j = 0; j < rows-1; j++) {
			for(int i = 0 ; i < cols-1; i++){
                of << "8" << endl;
			}
		} 
	

	}
	if(m_host->get_matrix_dimensions().size() == 3)	{
		int cols =m_host->get_matrix_dimensions().at(0);
		int rows = m_host->get_matrix_dimensions().at(1);
		int matrices = m_host->get_matrix_dimensions().at(2);
        of << "CELLS " << (cols-1)*(rows-1)*(matrices-1) <<" " << (cols-1)*(rows-1)*(matrices-1)*9 << endl;
		for(int k = 0; k < matrices-1; k++){		
			for(int j = 0; j < rows-1; j++) {
				for(int i = 0 ; i < cols-1; i++){
					//of << "8 " << i + j*cols+k*(cols*rows) << " " << i+1+j*cols+k*(cols*rows) << " " << i+1+(j+1)*cols+k*(cols*rows) << " " << i+(j+1)*cols+k*(cols*rows);
					//of << " " << i + j*cols+(k+1)*(cols*rows) << " " << i+1+j*cols+(k+1)*(cols*rows) << " " << i+1+(j+1)*cols+(k+1)*(cols*rows) << " " << i+(j+1)*cols+(k+1)*(cols*rows);
					of << "8 "  << i + j*cols+k*(cols*rows) << " " << i+1+j*cols+k*(cols*rows) << " " << i + j*cols+(k+1)*(cols*rows) << " " << i+1+j*cols+(k+1)*(cols*rows); 
					of << " " <<i+(j+1)*cols+k*(cols*rows) << " " << i+1+(j+1)*cols+k*(cols*rows) << " " << i+(j+1)*cols+(k+1)*(cols*rows)<< " " << i+1+(j+1)*cols+(k+1)*(cols*rows); 
                    of << endl;
				}
			} 
		}	
        of << "CELL_TYPES " << (cols-1)*(rows-1)*(matrices-1) << endl;
		for(int k = 0; k < matrices-1; k++){		
			for(int j = 0; j < rows-1; j++) {
				for(int i = 0 ; i < cols-1; i++){
                    of << "11" << endl;
				}
			} 
		}	

	}
    of << endl;
	delete m_host;
	delete it;

	//now write vector data
    of << "POINT_DATA " << num_entries << endl;
	if(dim == 1) {
        of << "SCALARS scalar_data DOUBLE 1" << endl;
        of << "LOOKUP_TABLE default" << endl;
	}
	else {
        of << "VECTORS vector_data DOUBLE" << endl;
	}
    vector<matrix_host*> matrices;
    vector<matrix_host_iterator*> iterators;
	for(int i = 0; i < dim; i++) {
        matrix_host* matrix = new matrix_host(folder->get_matrix(i), matrix_host::Copy);
		iterators.push_back(matrix_host_iterator::create_iterator(matrix));
		matrices.push_back(matrix);
	}
	of.setf(ios::showpos);
	for(int k = 0; k < num_entries; k++) {
        for(size_t i = 0; i < 3; i++) {
			if(iterators.size() > i){
				if(iterators[i]->has_next()){
					CUDA_FLOAT entry = iterators[i]->next();
					if(type == 1)
						of << entry.x << " ";
					else if(type == 0)
						of << entry.y << " ";
					else
						of << sqrt(entry.x*entry.x + entry.y*entry.y) << " ";
				} else {
					of << "0.0" << " ";
				}
			}else {
				//of << "0.0" << " ";
			}
		}
        of << endl;
	}
    of << endl;

	//free memory
	for(int i = 0; i < dim; i++) {
		delete matrices[i];
		delete iterators[i];
	}
	matrices.clear();
	iterators.clear();

	of.close();
	
}

void matrix_folder_writer::write_gnuplot_file(string filename, matrix_folder* folder){
    
    
	int dim = folder->get_dimension();
    
	for(int i = 0; i < dim; i++) {
        
        //create the right filename 
        string current_filename(filename);
        stringstream out;
        out << i;
        current_filename.append(out.str());
        
        // write the file
		write_gnuplot_file(current_filename, folder, i);
	}
    
}

void matrix_folder_writer::write_gnuplot_file(string filename, matrix_folder* folder, int matrix_i){
    
    
	int folder_dim = folder->get_dimension();
    if((matrix_i < 0) || (matrix_i >= folder_dim)) {
        EXIT_ERROR("Matrix index not found in folder.");
    }
    
    //...open the file and check if opened
    ofstream of(filename.c_str());
    if(!of.is_open()) {
        cerr << "Open file failed:" << filename << endl;
    }
    of.precision(10);
    of.setf(ios::scientific);
    
    //copy data to host
    matrix_host* m_host = new matrix_host(folder->get_matrix(matrix_i), matrix_host::Copy);
    vector<int> m_host_dim = m_host->get_matrix_dimensions();
    
    // write gnuplot file header
    of << "# Matrix " << matrix_i << " of size";
    for (unsigned int i = 0; i < m_host_dim.size(); ++i) {
        of << (i==0)?' ':'x' << m_host_dim.at(i);
    }
    of << '\n';
    of << "# Data ordering: x,y,.... , real(data), imag(data)" << endl;
    
    //write the information to file => column (x) / row (y) / matrix (z)
    matrix_host_iterator* it = matrix_host_iterator::create_iterator(m_host);
    while(it->has_next()) {
        CUDA_FLOAT entry = it->next();
        vector<int> indices = it->get_current_indices();
        for(size_t k = 0; k < indices.size(); k++)
            of << indices.at(k) << " ";
        of << entry.x  << " " << entry.y << endl;
    }
    
    //free tmp mem
    of.close();
    delete it;
    delete m_host;	
}

void matrix_folder_writer::write_binary_3d_layer(string filename, matrix_folder_real* folder, const char *description, const int layer) {

    // Check input matrix sizes
    // Check folder size (matrix 0 must be there)
    if(folder->get_dimension() < 1) {
        cerr << "write_binary_3d_layer: Empty folder" << endl;
        return;
    }
    // Check matrix dimension (must be 2 or 3)
    vector<int> dim = folder->get_matrix(0)->get_matrix_dimension();
    int dimsize = dim.size();
    if(dimsize < 2 || dimsize > 3) {
        cerr << "write_binary_3d_layer: Matrix dimension is not 2 or 3" << endl;
        return;
    }
    // Check if layer is there
    if((layer < 0) || (dimsize == 2 && layer != 0) || (dimsize == 3 && layer > dim.at(2))) {
        cerr << "write_binary_3d_layer: Layer "<<layer<<" not available" << endl;
        return;
    }

    // Copy matrix data from device to host
    matrix_host_real *current_host_copy = new matrix_host_real(folder->get_matrix(0), matrix_host_real::Copy);

    // Open the output file
    ofstream of(filename.c_str(), ios::binary);
    if(!of.is_open()) {
        cerr << "write_binary_3d_layer: Open file failed: " << filename << endl;
        return;
    }

    // Write description to file
    int len = strlen(description)+1;
    of.write((char*) &len, sizeof(int));
    of.write(description, len*sizeof(char));

    // Write information about the size of the matrix
    int dim_x = current_host_copy->get_matrix_size(0);
    int dim_y = current_host_copy->get_matrix_size(1);
    of.write((char*) &dim_x, sizeof(int));
    of.write((char*) &dim_y, sizeof(int));

    // Write out the data
    CUDA_FLOAT_REAL* data = current_host_copy->get_data();
    // Convert data to float
    for(int i = 0; i < dim_x*dim_y; ++i) {
        float dat = data[layer*dim_x*dim_y + i];
        of.write((char*) &dat, sizeof(float));
    }
    //of.write((char*) (data + layer*dim_x*dim_y), sizeof(CUDA_FLOAT) * dim_x*dim_y);

    // Delete host copy of data matrix
    delete current_host_copy;

    //close file
    of.close();
}

void matrix_folder_writer::convert_binary_3d_layer(const char* filename_in, const char* filename_out) {


    // Open the input file
    ifstream infile(filename_in, ios::in | ios::binary);
    if(!infile.is_open()) {
        cerr << "convert_binary_3d_layer: Open file failed: " << filename_in << endl;
        return;
    }

    // Open the output file
    ofstream of(filename_out);
    if(!of.is_open()) {
        cerr << "convert_binary_3d_layer: Open file failed: " << filename_out << endl;
        return;
    }

    // Read description from file
    int len = 0;
    infile.read((char*) &len, sizeof(int));
    if(len < 0 || len > 1024) {
        cerr << "convert_binary_3d_layer: Read failed: len = " << len << endl;
        return;
    }

    char* description = new char[len];
    infile.read(description, sizeof(char)*len);

    // Read information about the size of the matrix
    int dim_x = 0;
    int dim_y = 0;
    infile.read((char*) &dim_x, sizeof(int));
    infile.read((char*) &dim_y, sizeof(int));
    if(dim_x < 1 || dim_x > 32000 || dim_y < 1 || dim_y > 32000) {
        cerr << "convert_binary_3d_layer: Read failed: dim_x dim_y = " << dim_x << " " << dim_y << endl;
        return;
    }

    // Read in the data
    float* data = new float[dim_x*dim_y];
    infile.read((char*) data, sizeof(float) * dim_x*dim_y);

    // Read complete
    cout << "Convert " << dim_x << "x" << dim_y << " floats with description " << description << endl;

    // Write out the data
    of << "# " << description << endl;
    of << "# data ordering: x,y,.... ,real(data_x)" << endl;
    of.precision(4);
    of.setf(ios::scientific);

    for(int y = 0; y < dim_y; ++y) {
        of << '\n';
        for(int x = 0; x < dim_x; ++x) {
            of << x << '\t' << y << '\t' << data[x+y*dim_x] << '\n';
        }
    }

    // Delete memory
    delete [] description;
    delete [] data;

    //close file
    infile.close();
    of.close();
}

void matrix_folder_writer::write_gnuplot_3d_layer(string filename, matrix_folder_real* folder, const char *description, const int layer) {

    return write_gnuplot_vector_file_2d(filename, folder, description, 2, layer);
    /*//...open the file and check if opened
    ofstream of(filename.c_str());
    if(!of.is_open()) {
        cerr << "Open file failed: " << filename << endl;
        return;
    }

    // Move data to host
    int dim = folder->get_dimension();
    vector<matrix_host_real*> matrices;
    for(int k = 0; k < dim; k++) {
        matrices.push_back(new matrix_host_real(folder->get_matrix(k), matrix_host_real::Copy));
    }

    of << "#" << description << endl;
    of << "#data ordering: x,y,.... ,real(data_x)" << endl;
    of.precision(4);
    of.setf(ios::scientific);

    //write the data
    matrix_host_real_iterator* m_iterator = matrix_host_real_iterator::create_iterator(matrices[0]);
    while(m_iterator->has_next()) {
        m_iterator->next();
        vector<int> current_indices = m_iterator->get_current_indices();

        if(current_indices.at(2) == layer) {

            // Insert blank line to separate data blocks
            if(current_indices.at(0) == 0) {
                of << endl;
            }

            for(size_t k = 0; k < current_indices.size(); k++) {
                of << current_indices[k] << '\t';
            }
            for(int k = 0; k < dim; k++) {
                CUDA_FLOAT_REAL entry = matrices[k]->get_entry(current_indices);
                of << entry  << '\t';
            }
            of << endl;
        }
    }

    //close file
    of.close();

    //free memory
    for(int k = 0; k < dim; k++) {
        delete matrices[k];
    }*/
}

void matrix_folder_writer::write_gnuplot_vector_file(string filename, matrix_folder* folder) {
    cout << "Write vector file " << filename << endl;

    //...open the file and check if opened
    ofstream of(filename.c_str());
    if(!of.is_open()) {
        cerr << "Open file failed: " << filename << endl;
        return;
    }

    // Move data to host
    int dim = folder->get_dimension();
    vector<matrix_host*> matrices;
    for(int k = 0; k < dim; k++) {
        matrices.push_back(new matrix_host(folder->get_matrix(k), matrix_host::Copy));
    }

    of << "#data ordering: x,y,.... ,real(data_x),imag(data_x),real(data_y),imag(data_y),..." << endl;
    of.precision(5);
    of.setf(ios::scientific);

    //write the data
    matrix_host_iterator* m_iterator = matrix_host_iterator::create_iterator(matrices[0]);
    while(m_iterator->has_next()) {
        m_iterator->next();
        vector<int> current_indices = m_iterator->get_current_indices();

        // Insert blank line to separate data blocks
        if(current_indices.at(0) == 0) {
            of << endl;
        }

        for(size_t k = 0; k < current_indices.size(); k++) {
            of << current_indices[k] << '\t';
        }
        for(int k = 0; k < dim; k++) {
            CUDA_FLOAT entry = matrices[k]->get_entry(current_indices);
            of << entry.x  << '\t' << entry.y << '\t';
        }
        of << endl;
    }

    //close file
    of.close();

    //free memory
    for(int k = 0; k < dim; k++) {
        delete matrices[k];
    }
}

// Write file with (i_x, i_y, i_z, M_0(x,y,z), M_1(x,y,z),...)
void matrix_folder_writer::write_gnuplot_vector_file(string filename, matrix_folder_real* folder) {

    //...open the file and check if opened
    ofstream of(filename.c_str());
    if(!of.is_open()) {
        cerr << "Open file failed: " << filename << endl;
        return;
    }

    // Move data to host
    int dim = folder->get_dimension();
    vector<matrix_host_real*> matrices;
    for(int k = 0; k < dim; k++) {
        matrices.push_back(new matrix_host_real(folder->get_matrix(k), matrix_host_real::Copy));
    }

    of << "#data ordering: x,y,.... ,real(data_x),real(data_y),..." << endl;
    of.precision(5);
    of.setf(ios::scientific);

    //write the data
    matrix_host_real_iterator* m_iterator = matrix_host_real_iterator::create_iterator(matrices[0]);
    while(m_iterator->has_next()) {
        m_iterator->next();
        vector<int> current_indices = m_iterator->get_current_indices();

        // Insert blank line to separate data blocks
        if(current_indices.at(0) == 0) {
            of << endl;
        }
        for(size_t k = 0; k < current_indices.size(); k++) {
            of << current_indices[k] << '\t';
        }
        for(int k = 0; k < dim; k++) {
            CUDA_FLOAT_REAL entry = matrices[k]->get_entry(current_indices);
            of << entry << '\t';
        }
        of << endl;
    }

    //close file
    of.close();

    //free memory
    for(int k = 0; k < dim; k++) {
        delete matrices[k];
    }
}

// Write file with (i_x, i_y, i_z, M_0(x,y,z), M_1(x,y,z),...)
void matrix_folder_writer::write_gnuplot_vector_file_2d(string filename, matrix_folder_real* folder,
                                                        const char* description,
                                                        int select_dim, int select_val) {

    //...open the file and check if opened
    ofstream of(filename.c_str());
    if(!of.is_open()) {
        cerr << "Open file failed: " << filename << endl;
        return;
    }

    // Move data to host
    int dim = folder->get_dimension();
    vector<matrix_host_real*> matrices;
    for(int k = 0; k < dim; k++) {
        matrices.push_back(new matrix_host_real(folder->get_matrix(k), matrix_host_real::Copy));
    }

    of << "#" << description << '\n';
    of << "#data ordering: x,y,.... ,real(data_x),real(data_y),..." << endl;
    of.precision(5);
    of.setf(ios::scientific);

    vector<int> last_indices = matrices[0]->get_matrix_dimensions();

    //write the data
    matrix_host_real_iterator* m_iterator = matrix_host_real_iterator::create_iterator(matrices[0]);
    while(m_iterator->has_next()) {
        m_iterator->next();
        vector<int> current_indices = m_iterator->get_current_indices();

        // Skip lines that are not selected
        if(((unsigned int)select_dim < current_indices.size())
                && (current_indices.at(select_dim) != select_val)) {
            continue;
        }

        // Insert index
        for(size_t k = 0; k < current_indices.size(); k++) {
            of << current_indices[k] << '\t';
        }

        // Insert values
        for(int k = 0; k < dim; k++) {
            of << matrices[k]->get_entry(current_indices) << '\t';
        }

        // New line
        of << '\n';

        // Insert blank line to separate data blocks
        // One blank line is inserted for every index that jumps down
        for(size_t k = 0; k < current_indices.size(); k++) {
            if((k != (unsigned int)select_dim) && (current_indices.at(k) < last_indices.at(k))) {
                of << '\n';
            }
            last_indices.at(k) = current_indices.at(k);
        }
    }

    //close file
    of.close();

    //free memory
    for(int k = 0; k < dim; k++) {
        delete matrices[k];
    }
}


void matrix_folder_writer::write_gnuplot_particle_file(string filename, matrix_folder* folder, int starttime, App as,
                                                       int write_interval, double dt){
    // Copy data from device to host
    const int dim = folder->get_dimension();
    vector<matrix_host*> matrices;
	for(int k = 0; k < dim; k++) {
        matrices.push_back(new matrix_host(folder->get_matrix(k), matrix_host::Copy));
    }

    //...open the file and check if opened
    ios_base::openmode mode = ios_base::out;
    if((starttime != 0) && (as == append)) {
        mode = ios_base::app;
    } else {
        as = noappend;
    }
    ofstream of(filename.c_str(), mode);
    if(of.fail()) {
        EXIT_ERROR("not able to open output stream for particle positions!");
	}

    if(as == noappend) {
        of << "#data ordering: timestep,x of part. 1, y of part. 1, z of part.1 , x of part. 2" << endl;
    }
    of.precision(10);
    //of.setf(ios::scientific);

	//write the data
	matrix_host* x = matrices.at(0);
	matrix_host* y = matrices.at(1);
	matrix_host* z = matrices.at(2);

    //...loop over number of timesteps
	for(int t = 0; t < x->get_matrix_size(1); t++) {
        if((starttime + t) % write_interval == 0) {
            of << (starttime + t) * dt;
            //...loop over number of particles
            for(int i = 0; i < x->get_matrix_size(0); i++){
                int current_index = t*x->get_matrix_size(0) + i;
                of << '\t' << x->get_data()[current_index].x;
                of << '\t' << y->get_data()[current_index].x;
                of << '\t' << z->get_data()[current_index].x;
            }
            of << endl;
        }
	}
	//close file
    of.close();

	//free memory
	for(int k = 0; k < dim; k++) {
		delete matrices[k];
	}
    matrices.clear();
}




