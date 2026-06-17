// Computes time difference of two time periods
// Time periods are entered by the user

#include <iostream>
using namespace std;

struct TIME 
{
    int seconds;
    int minutes;
    int hours;
};

// computeTimeDifference() function calculates the difference between the time periods
void computeTimeDifference(struct TIME, struct TIME, struct TIME *);

int main() 
{    
    struct TIME t1, t2, difference;

    cout << "Enter start time." << endl;
    cout << "Enter hours, minutes and seconds respectively: " << endl;
    cin >> t1.hours >> t1.minutes >> t1.seconds;

    cout << "Enter stop time." << endl;
    cout << "Enter hours, minutes and seconds respectively: " << endl;
    cin >> t2.hours >> t2.minutes >> t2.seconds;

    computeTimeDifference(t1, t2, &difference);

    cout << endl << "TIME DIFFERENCE: " << t1.hours << ":" << t1.minutes << ":" << t1.seconds;
    cout << " - " << t2.hours << ":" << t2.minutes << ":" << t2.seconds;
    cout << " = " << difference.hours << ":" << difference.minutes << ":" << difference.seconds << endl;

    return 0;
}

/** computeTimeDifference() function calculates the difference between the time periods
 * Input: TIME t1 is the start time, TIME t2 is the stop time
 * Output: TIME *difference is the pointer that holds the result
 * @return the function return none due to the function type "void"
 * 
 * Improvement:
 * - the function should check the case t1 > t2, in that case, the function type should 
 *   be changeed to "bool" for returning a boolean flag denoting if function performed 
 *   successfully!
 */
void computeTimeDifference(struct TIME t1, struct TIME t2, struct TIME *difference) 
{    
    if(t2.seconds > t1.seconds)
    {
        --t1.minutes;
        t1.seconds += 60;
    }

    difference->seconds = t1.seconds - t2.seconds;
    if(t2.minutes > t1.minutes)
    {
        --t1.hours;
        t1.minutes += 60;
    }
    difference->minutes = t1.minutes-t2.minutes;
    difference->hours = t1.hours-t2.hours;
}
