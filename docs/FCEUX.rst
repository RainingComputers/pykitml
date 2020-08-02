Creating NES bots using FCEUX emulator
======================================

FCEUX Server
------------

.. autoclass:: pykitml.FCEUXServer

    .. automethod:: __init__

    .. automethod:: start

    .. automethod:: frame_advance

    .. automethod:: get_joypad

    .. automethod:: set_joypad

    .. automethod:: quit

    .. automethod:: compile

    .. automethod:: exec

    .. automethod:: send

    .. automethod:: recv

    .. autoattribute:: info

Lua client script 
-----------------

This script has to be loaded into the emulator after
starting the server. (File > Load Lua Script)

**fceux_client.lua**

.. code-block:: lua

    local socket = require "socket"

    -- Edit to change
    ip = 'localhost'
    port = '1234'

    -- Table for holding lua code snippets from server
    func_table = {}

    -- Start connection with server
    s = socket.connect('localhost', '1234')

    -- Helper function to convert table to string
    function table_to_string(table)
        str = ''

        for key, value in pairs(table) do
            str = str .. tostring(key) .. ' ' .. tostring(value) .. ' ' 
        end

        return str
    end

    -- Helper function to split string into token
    function split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={}
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                table.insert(t, str)
        end
        return t
    end

    -- Helper function to send server message
    function send(msg)
        s:send(msg)
    end

    -- Helper function to receive message from server
    function recv(msg)
        local resp, err = s:receive('*l')
        return resp
    end

    -- Helper function that waits for ackoledgement from server
    function wait_for_ack()
        while (recv() ~= 'ACK') do end
    end

    -- Set the speed of the emulator
    emu.speedmode('normal') 

    -- Server info
    send('FCEUX Client '.._VERSION)
    wait_for_ack()

    -- Main loop
    while true do
        local resp = ''

        -- Log frame count
        fcount = string.format('%d', emu.framecount())
        send(fcount)

        -- Parse commands from server
        while (resp ~= 'CONT') do
            resp = recv()

            if(resp == 'JOYPAD') then
                local controller = joypad.read(1)
                send(table_to_string(controller))
            elseif(resp == 'SETJOYPAD') then
                local values = split(recv())
                joypad.set(1, {
                    up = (values[1]=='True'), down = (values[2]=='True'),
                    left = (values[3]=='True'), right = (values[4]=='True'),
                    A = (values[5]=='True'), B = (values[6]=='True'),
                    start = (values[7]=='True'), select = (values[8]=='True'),
                })
            elseif(resp == 'CODE') then
                local id = recv()
                local code = recv()
                func_table[id] = loadstring(code)
            elseif(resp == 'EXEC') then
                local id = recv()
                f = func_table[id]
                if(f) then f() end
            else
                break
            end 
        end

        emu.frameadvance() 
    end

Examples
--------

**Example bot to spam the 'A' button**

.. code-block:: python
    
    import pykitml as pk

    def on_frame(server, frame): 
        # Spam A and start button
        if(frame%10 < 5): server.set_joypad(A=True, start=True)
        else: server.set_joypad(A=False, start=False)    

        # Print joypad
        print(server.get_joypad())

        # Continue emulation
        server.frame_advance()

    # Intialize and start server
    server = pk.FCEUXServer(on_frame)
    print(server.info)
    server.start()

Start this script, then run the FCEUX emulator. Open any NES ROM 
and then load the lua client script. The bot will continuously spam
the A button.

**Example, executing lua code on emulator**

.. code-block:: python

    import datetime

    import pykitml as pk

    # Lua code that will run on the emulator
    # Arguments are received using recv() function.
    print_msg = '''
        msg = recv()
        emu.message(msg)
    '''

    def on_frame(server, frame): 
        # Toggle start button to start rounds
        if(frame%10 < 5): server.set_joypad(A=True, start=True)
        else: server.set_joypad(A=False, start=False)    

        # Print joypad
        print(server.get_joypad())

        # Print date time
        # Pass all arguments to lua code as *args
        server.exec(print_msg, str(datetime.datetime.now()))

        # Continue emulation
        server.frame_advance()

    # Intialize and start server
    server = pk.FCEUXServer(on_frame)
    print(server.info)

    # Pass whatever code you want to compile as *args
    # when starting server
    server.start(print_msg)

FCEUX Lua API
-------------
See 
https://tasvideos.github.io/fceux/web/help/fceux.html?LuaFunctionsList.html
for FCEUX lua API docs. *Note:* Comments will not work in the lua code.

