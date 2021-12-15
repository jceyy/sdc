#include <iostream>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <time.h>

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
			data = readData.GetNewest()->Data();
	    }
    }
};

int SimVisual(SDL_Window* window, SDL_Renderer* renderer)
{
	srand(time(NULL));

    bool quit_visual(false), isFullscreen(false);
    size_t Nx(0), Ny(0); 
    int pitch(0), pitchCst(1);
    unsigned int format;
    size_t previousNt(0);

    float tempMin(0.f), tempMax(0.f), tempRange(1.f);
    #ifdef TRACK_VALUES 
    float tempRms(0.f);
	#endif // TRACK_VALUES
	#ifdef TIMER
	size_t ticks(0), frameTime(0), ellapsed(0);
	#endif // TIMER
    FILE *cfgFluidFile = fopen("config.txt", "r");
    
    if (cfgFluidFile != nullptr) // read the data needed if the opening succeeded
    {
        int temp = fscanf(cfgFluidFile, "%lu %lu", &Ny, &Nx);
        fclose(cfgFluidFile);
    }
    else  // default if it was unable to open the configuration file (pointer == NULL)
    {
        Nx = DEFAULT_NX;
        Ny = DEFAULT_NY;
        cfgFluidFile = fopen("config.txt", "w");
        fprintf(cfgFluidFile, "%lu %lu", Ny, Nx);
        printf("Unable to load the fluid config file, setting default values Nx = %ld, Ny = %ld\n", Nx, Ny);
    }

    vector<vector<float>> temperature(Nx, vector<float>(Ny, 0.5f));
    
    // SDL RELATED VARIABLES
    Uint32* pixels = nullptr;
    Uint8 R(0), G(0), B(0);
    SDL_Texture *texture = nullptr; // create the texture on which the program will write the temperature field
    SDL_Event event; // variable handling the events
    //SDL_Rect pos; // position (in pixels): where to display texture
    SDL_Point cursor, window_sz;
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, (int)Nx, (int)Ny); // add null-test
    if (texture == nullptr)
    {
    	cerr << "Unable to create the main texture: " << SDL_GetError() << endl;
    	return -1;
    }
    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_NONE);
	SDL_GetWindowSize(window, &window_sz.x, &window_sz.y);

    // Network sending structure
    NetUserInput netUserInput;
    netUserInput.relPosX = netUserInput.relPosY = 0.5f;
    netUserInput.radius = 0.2f;
    netUserInput.temperature = 0.1f;
    netUserInput.mode = 0;
    
    NetSimInfo netSimInfo;
    
    // Network Setup

    //uPacket upac;
    DataFolder readData;

    const char* HostName = "localhost";

    uRequest::ImgType imgtype = uRequest::Temperature;
    RBCClient::ReqType reqNew = RBCClient::newReq;
    RBCClient::ReqType reqCnt = RBCClient::cntReq;
    DataFolderClient socket(DEFAULT_SERVER_PORT, HostName);
    //unsigned short resReq = 0;
    //unsigned int us = 5000;

    int fieldInteractionSelector = 1; // before: currentMode. It is sent to the simulation only when a mouse button is pressed. 1 for temperature, 2 for velocity, 0 for nothing

	
    while(!quit_visual)
    {
    	#ifdef TIMER
    	ticks = SDL_GetTicks();
    	#endif
        // Events handling loop
        while (SDL_PollEvent(&event)) // while there is still events to catch up
        {  
            switch (event.type) // test the different types of events
            {
            	#ifdef ALLOW_INTERACTION
                case SDL_MOUSEBUTTONDOWN: // Active: while clicking, change the data field

                    cursor = {event.button.x, event.button.y};
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
                	cursor = {event.motion.x, event.motion.y};
                    netUserInput.relPosX = cursor.x / (float)window_sz.x;
                    netUserInput.relPosY = cursor.y / (float)window_sz.y;
                break;
                
                case SDL_FINGERDOWN:
                case SDL_FINGERMOTION:
                	cursor = {(int)(event.tfinger.x * window_sz.x), (int)(event.tfinger.y * window_sz.y)};
                	netUserInput.relPosX = event.tfinger.x;
                	netUserInput.relPosY = event.tfinger.y;
                	printf("\nFinger pressure is %.2f\n", event.tfinger.pressure);
                break;
                #endif // ALLOW_INTERACTION

                case SDL_KEYDOWN: // if a key was pressed
                switch (event.key.keysym.sym)
                {
                    case SDLK_ESCAPE: // is the key pressed is escape
                    quit_visual = true;
                    goto cleanup; // quit
                    break;
                    
                    case SDLK_f:
                    if (isFullscreen)
                    {
                    	SDL_SetWindowFullscreen(window, 0);
                    	isFullscreen = false;
                    }
                    else
                    {
                    	SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN_DESKTOP);
                    	isFullscreen = true;
                    }
                    break;
                }
                break;

				case SDL_WINDOWEVENT:
					if (event.window.event == SDL_WINDOWEVENT_RESIZED)
					{
						window_sz.x = event.window.data1;
						window_sz.y = event.window.data2;
					}
				break;
                
                case SDL_QUIT: // if the user wants to quit (Alt + F4 or clicked on "close")
                goto cleanup;
                break;
            }
        }// End of events loop        
		#ifdef TIMER
		ellapsed = SDL_GetTicks() - ticks;
    	ticks = SDL_GetTicks();
    	frameTime += ellapsed;
    	printf("Events handled in %lu ms\t", ellapsed);
    	#endif
        // Network communication part
		socket.Receive(&netUserInput, &readData, reqNew, imgtype, Nx);
		if(readData.GetNewest() != 0) // get the newest / best image
		{
			netSimInfo.update(readData, false);
			if (netSimInfo.Nt == previousNt) ;//goto receive;
			else previousNt = netSimInfo.Nt;

			while (readData.GetNewest()->Completeness() < 1.0f)
			{
				socket.Receive(&netUserInput, &readData, reqCnt, imgtype, Nx);
			}
			#ifdef TIMER
			ellapsed = SDL_GetTicks() - ticks;
			ticks = SDL_GetTicks();
			frameTime += ellapsed;
			printf("Network handled in %lu ms\t", ellapsed);
			#endif

			size_t j(0), k(0);
			tempMin = tempMax = 0.f;

			for (size_t i(0); i < netSimInfo.dataSz; i++)
			{
			    j = i%Nx;
			    //if (netSimInfo.data[i] != 0.0f || true) // use "true" to ignore 0.0 values
			    //{
			        temperature[j][k] = netSimInfo.data[i];
			        #ifdef TRACK_VALUES
			        tempRms += netSimInfo.data[i] * netSimInfo.data[i];
			        #endif // TRACK_VALUES
			    //}

			    if (temperature[j][k] < tempMin)
			    	tempMin = temperature[j][k];
			    else if (temperature[j][k] > tempMax)
			    	tempMax = temperature[j][k];

			    if (j == 0 && i > 0) k++;
			}
			tempRange = tempMax - tempMin;
			#ifdef TIMER
			ellapsed = SDL_GetTicks() - ticks;
			ticks = SDL_GetTicks();
			frameTime += ellapsed;
			printf("Temperature field update handled in %lu ms\t", ellapsed);
			#endif
			//printf("Nt = %lu, (Nx, Ny, clx, cly) = (%lu, %lu, %.1f, %.1f), (Ra, Pr, eta) = (%.2f, %.2f, %.2f), dt = %.3f, ",
			//netSimInfo.Nt, netSimInfo.Nx, netSimInfo.Ny, netSimInfo.clx, netSimInfo.cly, netSimInfo.Ra, netSimInfo.Pr, netSimInfo.eta, netSimInfo.dt);
			//printf("data size = %lu, Trms = %.4f, (Tmin, Tmax) = (%.4f, %.4f)\n", netSimInfo.dataSz, sqrt(_temp_average), minCoeff, maxCoeff);
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

        for (size_t i(0); i < Nx; i++) // Creating texture
        {
            for (size_t j(0); j < Ny; j++)
            {
				// update the value of the pixel color under the form of an 24 bit number 8 bits for each color*/
				grayScale1(tempMin, tempRange, temperature[i][j], &R, &G, &B); // default choice
				pixels[j * pitchCst + i] =  R<<16 | G<<8 | B;
            }
        }
        SDL_UnlockTexture(texture); // stop writing on the texture and make it available again for display
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
        
        #ifdef TIMER
        ellapsed = SDL_GetTicks() - ticks;
    	ticks = SDL_GetTicks();
    	frameTime += ellapsed;
		printf("Texture preparation handled in %lu ms\t", ellapsed);
		printf("Total frametime of %lu ms (%.2f fps)\t", frameTime, 1000.f/frameTime);
		frameTime = 0;
		#endif
		#ifdef TRACK_VALUES
		tempRms = sqrt(tempRms/(float)netSimInfo.dataSz); // mean square value: divide by Nx*Ny, or data size
		printf("Nt = %lu, t = %.3f, dt = %.3f, (Tmin, Tmax) = (%.4f, %.4f), Trms = %.4f", netSimInfo.Nt, netSimInfo.dt * netSimInfo.Nt, netSimInfo.dt, tempMin, tempMax, tempRms);
		tempRms = 0;
		#endif // TRACK_VALUES
		#if defined TIMER || defined TRACK_VALUES
		cout << endl;
		#endif
	}

    cleanup: // 3D array cleanup
    cout << "Quitting..." << endl;
	SDL_DestroyTexture(texture); // free the memory occupied by the texture    

    return 0;
}


int run(SDL_Window* window, SDL_Renderer* renderer)
{
	return SimVisual(window, renderer);
}

