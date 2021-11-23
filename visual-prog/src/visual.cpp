#include <iostream>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#include "coloring.h"
#include "defines.h"
#include "network/datastruct.h"
#include "network/server_writer.h"
#include "network/serverclient.h"
#include "network/datafile.h"

using namespace std;

// Structure of data that the visualization gets from the simulation
class NetSimInfo
{
public:
	NetSimInfo() : np(0), nr(0), Nx(0), Ny(0), Nt(0), dataSz(0), dt(0.f), Ra(0.f),
	Pr(0.f), eta(0.f), clx(0.f), cly(0.f), clz(0.f), data(nullptr) {}
    
    short np, nr;
    size_t Nx, Ny, Nt, dataSz;
    float dt, Ra, Pr, eta;
    float clx, cly, clz;
    float* data;
    
    void update(DataFolder& readData, bool getBest = false)
    {
		if (getBest)
		{
			np = readData.GetBest()->NumPackets();
			nr = readData.GetBest()->CountPackets();
			Ra = readData.GetBest()->H().Ra;
			dt = readData.GetBest()->H().dt;
			eta = readData.GetBest()->H().eta;
			Pr = readData.GetBest()->H().Pr;
			Nx = readData.GetBest()->H().nmx;
			Ny = readData.GetBest()->H().nmy;
			Nt = readData.GetBest()->Nt();
			clx = readData.GetBest()->H().clx;
			cly = readData.GetBest()->H().cly;
			clz = readData.GetBest()->H().clz;
			dataSz = readData.GetBest()->Size();
			data = readData.GetBest()->Data();
	    }
	    else
	    {
			np = readData.GetNewest()->NumPackets();
			nr = readData.GetNewest()->CountPackets();
			Ra = readData.GetNewest()->H().Ra;
			dt = readData.GetNewest()->H().dt;
			eta = readData.GetNewest()->H().eta;
			Pr = readData.GetNewest()->H().Pr;
			Nx = readData.GetNewest()->H().nmx;
			Ny = readData.GetNewest()->H().nmy;
			Nt = readData.GetNewest()->Nt();
			clx = readData.GetNewest()->H().clx;
			cly = readData.GetNewest()->H().cly;
			clz = readData.GetNewest()->H().clz;
			dataSz = readData.GetNewest()->Size();
			data = readData.GetBest()->Data();
	    }
    }
};


int SimVisual(SDL_Window* window, SDL_Renderer* renderer)
{
    // define variables for the code
    bool quit_visual(false);
    size_t Nx(0), Ny(0);//, Nz(0);
    int COLORATION=0;
    int pitch = 0, pitchCst=1;
    unsigned int format;
    int previousNt(0);

    float minCoeff=0.0, maxCoeff=1.0;

    // create a pointer to the file we want to read (read only, "r") from
    FILE *cfgFluidFile = nullptr;
    cfgFluidFile = fopen("config/fluid_config.txt", "r");
    
    if (cfgFluidFile != nullptr)// read the data needed if the opening succeeded
    {
        fscanf(cfgFluidFile, "%lu %lu", &Nx, &Ny);
        fclose(cfgFluidFile);
    }
    else  // default if it was unable to open the configuration file (pointer == NULL)
    {
        Nx = DEFAULT_NX;
        Ny = DEFAULT_NY;
        printf("Unable to load the fluid config file, setting default values Nx = %ld, Ny = %ld\n", Nx, Ny);
    }

    vector<vector<float>> temperature(Nx, vector<float>(Ny, 0.5f));
    
    // SDL RELATED VARIABLES
    Uint32* pixels = nullptr;
    Uint8 R(0), G(0), B(0);
    SDL_Texture *texture = nullptr; // create the texture on which the program will write the temperature field
    SDL_Event event; // variable handling the events
    SDL_Rect pos; // position (in pixels): where to display texture
    SDL_Point mouse = {event.button.x, event.button.y};
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, (int)Nx, (int)Ny); // add null-test
    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_NONE);


    // Network sending structure
    NetUserInput netUserInput;
    netUserInput.relPosX = netUserInput.relPosY = 0.5f;
    netUserInput.radius = 0.1f;
    netUserInput.temperature = 1.0f;
    netUserInput.mode = 0;
    
    NetSimInfo netSimInfo;
    
    // Network Setup
    int cpt1(1);

    uPacket upac;
    DataFolder readData;

    const char* HostName = "localhost";

    uRequest::ImgType imgtype = uRequest::Temperature;
    RBCClient::ReqType reqNew = RBCClient::newReq;
    RBCClient::ReqType reqCnt = RBCClient::cntReq;
    DataFolderClient socket(DEFAULT_SERVER_PORT, HostName, cpt1);
    //unsigned short resReq = 0;
    //ofstream f_out ("Fichier.txt");
    //unsigned int us = 5000;

    int fieldInteractionSelector = 1; // before: currentMode. It is sent to the simulation only when a mouse button is pressed. 1 for temperature, 2 for velocity, 0 for nothing


    while(!quit_visual)
    {
        // Events handling loop
        while (SDL_PollEvent(&event)) // while there is still events to catch up
        {  
            switch (event.type) // test the different types of events
            {
            	#if ALLOW_INTERACTION == true
                case SDL_MOUSEBUTTONDOWN: // Active: while clicking, change the data field

                    mouse = {event.button.x, event.button.y};
                    //cout << "Mouse button down" << endl; 
                    // Check the collisions
                    /*if (Coll_2AABBs(mouse, setting_position)) // if the moused clicked on the settings icon
                    {
                        isDisplayingPannel = true;
                    }
                    else
                    {
                        netUserInput.mode = currentMode;
                    }
                    netUserInput.radius = radius_value;
                    netUserInput.temp = temperature_value;*/
                    netUserInput.mode = fieldInteractionSelector;
                break;

                case SDL_MOUSEBUTTONUP:
                	//cout << "Mouse button up" << endl; 
                    netUserInput.mode = 0; // Idle: don't change the temperature field
                break;

                case SDL_MOUSEMOTION:
                	mouse = {event.motion.x, event.motion.y};
                    netUserInput.relPosX = mouse.x / (float)1280;
                    netUserInput.relPosY = mouse.y / (float)720;
                break;
                #endif // ALLOW_INTERACTION

                case SDL_KEYDOWN: // if a key was pressed
                switch (event.key.keysym.sym)
                {
                    case SDLK_ESCAPE: // is the key pressed is escape
                    quit_visual = true;
                    goto cleanup; // quit
                    break;
                }
                break;
                
                case SDL_QUIT: // if a key was pressed
                goto cleanup;
                break;
            }
        }// End of events loop        


        // Network communication part
        socket.Receive(&netUserInput, &readData, reqNew, imgtype, Nx);
        /*if(netSimInfo.nr > 0 || true)
        {
            for(int volatile e(0); e<400; e++)
            {
                socket.Receive(&netUserInput, &readData, reqCnt, imgtype, Nx); // receive data
                //cout << "nr = " << (int)nr << endl;
            }
            //cout << "nr = " << (int)nr << endl;
		//cout << "Socket received" << endl;
        }*/
        /*volatile int e(0);
        do
        {
        	socket.Receive(&netUserInput, &readData, reqCnt, imgtype, Nx); e++;
        	if(readData.GetBest() != 0 && readData.GetNewest()->CountPackets() <= 0)
        		break;
        }while(false && readData.GetNewest()->CountPackets() > 0 || e < 2 && true);*/

        //socket.Receive(&netUserInput, &readData, reqCnt, imgtype, Nx);
        
		if(readData.GetBest() != 0) // get the best image
		{
			/*auto tempv(readData.GetBest());
			while (tempv->NumPackets() > 0)
			{
				socket.Receive(&netUserInput, &readData, reqCnt, imgtype, Nx);
			}*/
			
			netSimInfo.update(readData, false);
			if (netSimInfo.Nt == previousNt)
				continue;
			else
				previousNt = netSimInfo.Nt;
				
			size_t j(0), k(0);
			float _temp_average(0.f); // for debugging only
			//minCoeff = maxCoeff = 0.f;
			for (size_t i(0); i < netSimInfo.dataSz; i++)
			{
			    j = i%Nx;
			    if (netSimInfo.data[i] != 0.0f || true) // use "true" to ignore 0.0 values
			    {
			        temperature[j][k] = netSimInfo.data[i];//*(pointerData + i);
			        //cout << netSimInfo.data[i] << " ";
			        _temp_average += temperature[j][k] * temperature[j][k];
			    }

			    if (temperature[j][k] < minCoeff)
			    	minCoeff = temperature[j][k];
			    else if (temperature[j][k] > maxCoeff)
			    	maxCoeff = temperature[j][k];

			    if (j == 0 && i > 0) k++;
			} //cout << endl;
			
			//_temp_average /= (float)netSimInfo.dataSz; // mean square value: divide by Nx*Ny, or data size
			cout << "Nt = " << netSimInfo.Nt << ", Nx*Ny = " << netSimInfo.dataSz << ", (Pr, Ra, eta) = (" << netSimInfo.Pr <<
			", " << netSimInfo.Ra << ", " << netSimInfo.eta << "), (clx, cly) = (" << netSimInfo.clx << ", " << netSimInfo.cly << 
			"), dt = " << netSimInfo.dt << ", (Nx, Ny) = (" << netSimInfo.Nx << ", " << netSimInfo.Ny << ")" <<
			", rms² value is " << (float)_temp_average << endl;
			//_temp_average = 0;
			
			/*volatile float temp_ = minCoeff;
			minCoeff=maxCoeff;
			maxCoeff=temp_; //*/
			
			//readData.hasChangedReset();
		}
		else
		{
			//cerr << "Unable to recover data from network\n";
		}

        // Update the texture
        if(SDL_QueryTexture(texture, &format, nullptr, nullptr, nullptr) != 0)
        {
            cerr << SDL_GetError() << endl;
            //goto cleanup;
        }

        // Lock the texture in order to get informations about it, and the write on it
        if (SDL_LockTexture(texture, nullptr, (void**)&pixels, &pitch))
        {
            cerr << SDL_GetError() << endl;
            //goto cleanup;
        }
        pitchCst = (pitch / sizeof(unsigned int));

        /*#if PRINT_INFOS==1
        timer_global.update(); // Measuring timing
        time = timer_global.getTime_millis();
        #endif*/

        for (size_t i(0); i < Nx; i++) // Creating texture
        {
            for (size_t j(0); j < Ny; j++)
            {
				/*if (COLORATION == 1) colorScale1(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B);
				else if (COLORATION == 2) colorScale2(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B);
				else if (COLORATION == 3) colorScale3(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B);
				else if (COLORATION == 4) colorScale4(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B);
				else if (COLORATION == 5) grayScale2(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B);
				else grayScale1(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B); // default choice
				// update the value of the pixel color under the form of an 24 bit number 8 bits for each color*/
				pixels[j * pitchCst + i] =  R<<16 | G<<8 | B;
				grayScale1(minCoeff, maxCoeff, temperature[i][j], &R, &G, &B); // default choice
				// cout << "(r, g, b) = (" << (int)R << ", " << (int)G << ", " << (int)B << ")" << endl;
            }
        }
        SDL_UnlockTexture(texture); // stop writing on the texture and make it available again for display
        pos = {0, 0, 1280, 720};//(int)Nx, (int)Ny};
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, &pos);
        SDL_RenderPresent(renderer);

        /*if (bool_infos) // if displaying the informations
        {
            factor=1.0f;
            posx0=50; posy0=200;
            nn=0; nnmax=0;

            // preparing the strings
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Ra : %d", netInData.Ra); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Eta : %.3f", netInData.eta); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Pr : %.3f", netInData.Pr); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "dt : %.3f", netInData.dt); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Cube length : %.3f x %.3f x %.3f", netInData.clx, netInData.cly, netInData.clz); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Iteration : %d", netInData.Nt); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Mouse position : %.3f  %.3f", netUserInput.posX, netUserInput.posY); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Mouse velocity : %.3f  %.3f", netUserInput.velX, netUserInput.velY); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Modification mode : %d (1 = temperature, 2 = speed)", currentMode); nnmax++;
            strcpy(text[nnmax], ""); sprintf(text[nnmax], "Coloration : %d", COLORATION); nnmax++;

        }*/
           
    }

    cleanup: // 3D array cleanup
	SDL_DestroyTexture(texture); // free the memory occupied by the texture    

    return 0;
}


int run(SDL_Window* window, SDL_Renderer* renderer)
{
	return SimVisual(window, renderer);
    /*SDL_SetRenderDrawColor(renderer_GLOBAL, 255, 255, 255, 255);
    SDL_RenderClear(renderer_GLOBAL);
    SDL_RenderPresent(renderer_GLOBAL);

    const int CST=1;
    PRL_MenuButton button1[CST]; // now obsolete class, to be replaced soon in PRL framework

    for (int i=0; i<CST; i++) // here CST is used for a more general case using more than 1 button
    {
        button1[i].setRenderer(renderer_GLOBAL);
        button1[i].setCenter(config_GLOBAL.renderResolution.x/2, config_GLOBAL.renderResolution.y/2);
        button1[i].loadIdleTexture("data/4K_start_button_idle.png");
        button1[i].loadSelecTexture("data/4K_start_button_selec.png");
        button1[i].setUse2States(true); // use idle and selected
    }

    SDL_Event event;
    int quit=0;

    while (!quit)
    {
        SDL_WaitEvent(&event); // wait for events and delay the program until an interruption occurs
        switch (event.type)
        {
        case SDL_QUIT:
            quit=1;
            return PRL_RETURN_QUIT;
            break;

        case SDL_WINDOWEVENT:
            if (event.window.type==SDL_WINDOWEVENT_CLOSE)
            {
                quit=1;
                return PRL_RETURN_QUIT;
            }
        break;

        case SDL_KEYDOWN:
            if (event.key.keysym.sym==SDLK_ESCAPE)
            {
                quit=1;
                return PRL_RETURN_QUIT;
            }
            else if (event.key.keysym.sym==SDLK_SPACE||event.key.keysym.sym==SDLK_RETURN)
            {
                if(SimVisual()==PRL_RETURN_QUIT) {quit=1; return PRL_RETURN_QUIT;}
            }
            break;

        case SDL_MOUSEMOTION:
            PRL_Point p;
            p.x=event.motion.x;
            p.y=event.motion.y;
            PRL_Rect r;
            for (int i=0; i<CST; i++)
            {
                r.x=button1[i].getCenter().x-button1[i].getSize().x/2.0;
                r.y=button1[i].getCenter().y-button1[i].getSize().y/2.0;
                r.w=button1[i].getSize().x;
                r.h=button1[i].getSize().y;

                button1[i].setIdle(!Coll_PointAABB(p, r));
                button1[i].update();
            }
            break;

        case SDL_MOUSEBUTTONDOWN:
            PRL_Point p2={event.button.x, event.button.y};
            PRL_Rect r2;
            for (int i=0; i<CST; i++)
            {
                r2.x=button1[i].getCenter().x-button1[i].getSize().x/2.0;
                r2.y=button1[i].getCenter().y-button1[i].getSize().y/2.0;
                r2.w=button1[i].getSize().x;
                r2.h=button1[i].getSize().y;
                if (Coll_PointAABB(p2, r2))
                {
                    if(SimVisual()==PRL_RETURN_QUIT) {quit=1; return PRL_RETURN_QUIT;}
                }
            }
            break;
        }

        SDL_RenderClear(renderer_GLOBAL);

        for (int i=0; i<CST; i++)
        {
            SDL_RenderCopy(renderer_GLOBAL, button1[i].getDispTexture(), NULL, button1[i].getDispPosition());
        }
        SDL_RenderPresent(renderer_GLOBAL);
    }*/
    //return 0;
}
